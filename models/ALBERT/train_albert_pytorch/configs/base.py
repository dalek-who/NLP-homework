from pathlib import Path
import socket
from datetime import datetime
import os

BASE_DIR = Path('.')

config = {
    'data_dir': BASE_DIR / 'dataset/sentiment_classification_raw',
    'log_dir': BASE_DIR / 'outputs/logs',
    'figure_dir': BASE_DIR / "outputs/figure",
    'outputs': BASE_DIR / 'outputs',
    'checkpoint_dir': BASE_DIR / "outputs/checkpoints",
    'result_dir': BASE_DIR / "outputs/result",
    'pred_dir': BASE_DIR / "outputs/predict",

    'bert_dir':BASE_DIR / 'pretrain/pytorch/albert_base_zh',
    'albert_config_path': BASE_DIR / 'configs/config.json',
    'albert_vocab_path': BASE_DIR / 'configs/vocab.txt'
}

def create_config(commit, model_size, load_checkpoints_dir=""):
    DATA_DIR = BASE_DIR / 'dataset/sentiment_classification/'
    OUTPUT_DIR = BASE_DIR / f"outputs/{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}_{model_size}__{commit}"
    CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
    my_config = {
        # inputs:
        #   model
        'bert_dir':  BASE_DIR / load_checkpoints_dir if load_checkpoints_dir else BASE_DIR / f'pretrain/pytorch/albert_{model_size}_zh' ,
        'albert_vocab_path': BASE_DIR / 'configs/vocab.txt',
        #   data
        'data_dir': DATA_DIR,
        'train_data_path': DATA_DIR / 'train.csv',
        'valid_data_path': DATA_DIR / 'valid.csv',
        'test_data_path': DATA_DIR / 'test.csv',
        'pred_data_path': DATA_DIR / 'pred.csv',

        # outputs:
        "output_dir": OUTPUT_DIR,
        "checkpoints_dir": CHECKPOINTS_DIR,
        "checkpoints_config": CHECKPOINTS_DIR / "config.json",
        "checkpoints_bin": CHECKPOINTS_DIR / "pytorch_model.bin",
        "predict_softmax": OUTPUT_DIR / "Test_Dataset_softmax.csv",
        "predict_result": OUTPUT_DIR / "Test_Dataset_Label.csv",
        "log": OUTPUT_DIR / "finetuning.log",
        "args": OUTPUT_DIR / "args.json",
        "config": OUTPUT_DIR / "config.json",
        "best_eval_metrics": OUTPUT_DIR / "best_eval_metrics.json",
        "scalars": OUTPUT_DIR / "all_scalars.json",
        "success_train": OUTPUT_DIR / "zzz_SUCCESS_train.txt",
        "success_predict": OUTPUT_DIR / "zzz_SUCCESS_predict.txt",

        # tensorboardX records:
        "step_train_loss": "tbX/step_train_loss",
        "step_learning_rate": "tbX/step_learning_rate",
        "epoch_loss": "tbX/epoch_loss",
        "epoch_acc": "tbX/epoch_acc",
    }
    return my_config
