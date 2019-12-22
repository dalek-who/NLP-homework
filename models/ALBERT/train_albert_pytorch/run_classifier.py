from __future__ import absolute_import, division, print_function
import argparse
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from model.file_utils import WEIGHTS_NAME, CONFIG_NAME
from model.modeling_albert import BertConfig
from model.optimization import AdamW, WarmupLinearSchedule
from common.tools import seed_everything
from common.tools import logger, init_logger
from configs.base import create_config
from model.modeling_albert import BertForSequenceClassification
from callback.progressbar import ProgressBar
# from lcqmc_progressor import BertProcessor
from ccf_processor import  BertProcessor
from common.metrics import Accuracy
from common.tools import AverageMeter
from pandas import DataFrame, Series
import os
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import json


def train(args, train_dataloader, valid_dataloader, train_dataloader_eval, metrics, model, config):
    """ Train the model """
    writer = SummaryWriter(config["output_dir"])
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    args.warmup_steps = t_total * args.warmup_proportion
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_acc = 0
    model.zero_grad()
    seed_everything(args.seed)
    for epoch in range(int(args.num_train_epochs)):
        tr_loss = AverageMeter()
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) if isinstance(t, torch.Tensor) else t for t in batch)
            seq_ids = batch[-1]
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss.update(loss.item(), n=1)
            pbar(step, info={"loss": loss.item()})
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 画图
                lr = scheduler.get_lr()[0]
                writer.add_scalars(config["step_train_loss"], {"train loss": loss.item()}, global_step)
                writer.add_scalars(config["step_learning_rate"], {"learning rate": lr}, global_step)
                # update
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        # train_log = {'loss': tr_loss.avg}
        train_log = evaluate(args, model, train_dataloader_eval, metrics)
        valid_log = evaluate(args, model, valid_dataloader, metrics)
        logs = {"train_loss": train_log["loss"], "train_acc": train_log["acc"],
                "valid_loss": valid_log["loss"], "valid_acc": valid_log["acc"],
                "epoch": epoch}

        writer.add_scalars(config["epoch_loss"], {"epoch_train_loss": logs["train_loss"],
                                                  "epoch_valid_loss": logs["valid_loss"]}, epoch)
        writer.add_scalars(config["epoch_acc"],  {"epoch_train_acc": logs["train_acc"],
                                                  "epoch_valid_acc": logs["valid_acc"]}, epoch)
        # logs = dict(train_log, **valid_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)

        if logs['valid_acc'] > best_acc:
            logger.info(f"\nEpoch {epoch}: valid_acc improved from {best_acc} to {logs['valid_acc']}")
            logger.info("save model to disk.")
            best_acc = logs['valid_acc']
            print("Valid Entity Score: ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # output_file = args.model_save_path
            output_file = config['checkpoints_dir']
            output_file.mkdir(exist_ok=True)
            output_model_file = output_file / WEIGHTS_NAME
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = output_file / CONFIG_NAME
            with open(str(output_config_file), 'w') as f:
                f.write(model_to_save.config.to_json_string())
            with open(config["best_eval_metrics"], "w") as fb:
                json.dump(logs, fb, indent=4)
    writer.export_scalars_to_json(config["scalars"])
    writer.close()

def evaluate(args, model, eval_dataloader, metrics):
    # Eval!
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = AverageMeter()
    metrics.reset()
    preds = []
    targets = []
    pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
    for bid, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) if isinstance(t, torch.Tensor) else t for t in batch)
        seq_ids = batch[-1]
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            eval_loss.update(loss.item(), n=batch[0].size()[0])
        preds.append(logits.cpu().detach())
        targets.append(inputs['labels'].cpu().detach())
        pbar(bid)
    preds = torch.cat(preds, dim=0).cpu().detach()
    targets = torch.cat(targets, dim=0).cpu().detach()
    metrics(preds, targets)
    eval_log = {"acc": metrics.value(),
                'loss': eval_loss.avg}
    return eval_log

def predict(args, model, pred_dataloader, config):
    # Predict (without compute metrics)
    # args.predict_save_path = config['pred_dir'] / f'{args.pred_dir_name}'
    # args.predict_save_path.mkdir(exist_ok=True)

    logger.info("  Num examples = %d", len(pred_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    seq_ids = []
    preds = []
    pbar = ProgressBar(n_total=len(pred_dataloader), desc='Predicting')
    for bid, batch in enumerate(pred_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) if isinstance(t, torch.Tensor) else t for t in batch)
        seq_ids += list(batch[-1])
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            ##############
            # writer = SummaryWriter(config["output_dir"])
            # ips = {k: v[[0], ...] for k, v in inputs.items()}
            # ops = model(**ips)
            # model_graph_inputs = (
            #     ips["input_ids"], ips["attention_mask"], ips["token_type_ids"], [1,2], [3,4], ips["labels"])
            # writer.add_graph(model, model_graph_inputs)
            # writer.close()
            ##############
            outputs = model(**inputs)
            loss, logits = outputs[:2]
        preds.append(logits.cpu().detach())
        pbar(bid)
    preds = torch.cat(preds, dim=0).cpu().detach()
    preds_label = torch.argmax(preds, dim=1)
    result_label = DataFrame(data={"id": Series(seq_ids), "label": Series(preds_label)})
    result_label.to_csv(config["predict_result"], index=False)

    preds_softmax = torch.softmax(preds, dim=1)
    result_softmax = DataFrame(data={"id": Series(seq_ids),
                                     "label_0": Series(preds_softmax[:, 0]),
                                     "label_1": Series(preds_softmax[:, 1]),
                                     "label_2": Series(preds_softmax[:, 2]) })
    result_softmax.to_csv(config["predict_softmax"], index=False)

    return result_label


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--arch", default='albert_xlarge', type=str)
    # parser.add_argument('--task_name', default='lcqmc', type=str)
    parser.add_argument("--train_max_seq_len", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_max_seq_len", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--share_type', default='all', type=str, choices=['all', 'attention', 'ffn', 'None'])
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    # parser.add_argument("--evaluate_during_training", action='store_true',
    #                     help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=int,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--train_data_num', type=int, default=None, help="Use a small number to test the full code")
    parser.add_argument('--eval_data_num', type=int, default=None, help="Use a small number to test the full code")
    parser.add_argument('--do_pred', action='store_true', help="Predict a dataset and do not evaluate accuracy")
    parser.add_argument('--model_size', type=str, default='large',
                        help="Which albert size to choose, could be: base, large, xlarge, xxlarge")
    parser.add_argument('--commit', type=str, default='', help="Current experiment's commit")
    parser.add_argument('--load_checkpoints_dir', type=str, default="",
                        help="Whether to use checkpoints to load model. If not given checkpoints, use un-fineturned albert")

    args = parser.parse_args()

    config = create_config(commit=args.commit, model_size=args.model_size, load_checkpoints_dir=args.load_checkpoints_dir)
    # args.model_save_path = config['checkpoints_dir']
    # args.model_save_path.mkdir()
    # os.makedirs(config["checkpoints_dir"])
    os.makedirs(config["output_dir"])

    with open(config["args"], "w") as fa, open(config["config"], "w") as fc:
        json.dump(vars(args), fa, indent=4)
        json.dump({k: str(v) for k,v in config.items()}, fc, indent=4)

    # Setudistant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    init_logger(log_file=config['log'])
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)
    # --------- data
    processor = BertProcessor(vocab_path=config['albert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    bert_config = BertConfig.from_pretrained(str(config['bert_dir'] / 'config.json'),
                                             share_type=args.share_type, num_labels=num_labels)

    logger.info("Training/evaluation parameters %s", args)
    metrics = Accuracy(topK=1)
    # Training
    if args.do_train:
        # train_data = processor.get_train(config['data_dir'] / "train.tsv")
        # train_examples = processor.create_examples(lines=train_data, example_type='train',
        #                                            cached_examples_file=config[
        #                                                                     'data_dir'] / f"cached_train_examples_{args.arch}")
        # todo: 划分数据集，合成train.csv, eval.csv, test. csv
        train_examples = processor.read_data_and_create_examples(
            example_type='train', cached_examples_file=config['data_dir'] / f"cached_train_examples_{args.model_size}",
            input_file=config['data_dir'] / "train.csv")
        train_examples = train_examples[:args.train_data_num] if args.train_data_num is not None else train_examples

        train_features = processor.create_features(examples=train_examples, max_seq_len=args.train_max_seq_len,
                                                   cached_features_file=config[
                                                                            'data_dir'] / "cached_train_features_{}_{}".format(
                                                       args.train_max_seq_len, args.model_size
                                                   ))
        train_features = train_features[:args.train_data_num] if args.train_data_num is not None else train_features

        train_dataset = processor.create_dataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        train_sampler_eval = SequentialSampler(train_dataset)
        train_dataloader_eval = DataLoader(train_dataset, sampler=train_sampler_eval, batch_size=args.eval_batch_size)

        # valid_data = processor.get_dev(config['data_dir'] / "dev.tsv")
        # valid_examples = processor.create_examples(lines=valid_data, example_type='valid',
        #                                            cached_examples_file=config[
        #                                                                     'data_dir'] / f"cached_valid_examples_{args.arch}")
        valid_examples = processor.read_data_and_create_examples(
            example_type='valid', cached_examples_file=config['data_dir'] / f"cached_valid_examples_{args.model_size}",
            input_file=config['data_dir'] / "valid.csv")
        valid_examples = valid_examples[:args.eval_data_num] if args.eval_data_num is not None else valid_examples

        valid_features = processor.create_features(examples=valid_examples, max_seq_len=args.eval_max_seq_len,
                                                   cached_features_file=config[
                                                                            'data_dir'] / "cached_valid_features_{}_{}".format(
                                                       args.eval_max_seq_len, args.model_size
                                                   ))
        valid_features = valid_features[:args.eval_data_num] if args.eval_data_num is not None else valid_features

        valid_dataset = processor.create_dataset(valid_features)
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

        model = BertForSequenceClassification.from_pretrained(config['bert_dir'], config=bert_config)
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        model.to(args.device)
        train(args, train_dataloader, valid_dataloader, train_dataloader_eval, metrics, model, config)
        # 打上戳表示训练完成
        config["success_train"].open("w").write("Train Success!!")

    # if args.do_test:
    #     model = BertForSequenceClassification.from_pretrained(args.model_save_path, config=bert_config)
    #     test_data = processor.get_train(config['data_dir'] / "test.tsv")
    #     test_examples = processor.create_examples(lines=test_data,
    #                                               example_type='test',
    #                                               cached_examples_file=config[
    #                                                                        'data_dir'] / f"cached_test_examples_{args.arch}")
    #     test_features = processor.create_features(examples=test_examples,
    #                                               max_seq_len=args.eval_max_seq_len,
    #                                               cached_features_file=config[
    #                                                                        'data_dir'] / "cached_test_features_{}_{}".format(
    #                                                   args.eval_max_seq_len, args.arch
    #                                               ))
    #     test_dataset = processor.create_dataset(test_features)
    #     test_sampler = SequentialSampler(test_dataset)
    #     test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
    #     # model = BertForSequenceClassification.from_pretrained(args.model_save_path, config=bert_config)
    #     model.to(args.device)
    #     test_log = evaluate(args, model, test_dataloader, metrics)
    #     print(test_log)

    if args.do_pred:
        pred_examples = processor.read_data_and_create_examples(
            example_type='predict', cached_examples_file=config['data_dir'] / f"cached_pred_examples_{args.model_size}",
            input_file=config['data_dir'] / "pred.csv")
        pred_examples = pred_examples[:args.eval_data_num] if args.eval_data_num is not None else pred_examples

        pred_features = processor.create_features(examples=pred_examples,
                                                  max_seq_len=args.eval_max_seq_len,
                                                  cached_features_file=config[
                                                                           'data_dir'] / "cached_pred_features_{}_{}".format(
                                                      args.eval_max_seq_len, args.model_size
                                                  ))
        pred_features = pred_features[:args.eval_data_num] if args.eval_data_num is not None else pred_features
        pred_dataset = processor.create_dataset(pred_features)
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.eval_batch_size)
        param_dir = config['checkpoints_dir'] if (config["checkpoints_dir"] / "pytorch_model.bin").exists() \
            else config['bert_dir']
        model = BertForSequenceClassification.from_pretrained(param_dir, config=bert_config)
        model.to(args.device)
        # todo
        predict(args, model, pred_dataloader, config)
        # 打上戳表示预测完成
        config["success_predict"].open("w").write("Predict Success!!")
    config["output_dir"].rename(config["output_dir"].parent / ("success-" + config["output_dir"].name))


if __name__ == "__main__":
    main()
