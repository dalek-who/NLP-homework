import torch
import csv
from models.ALBERT.train_albert_pytorch.callback.progressbar import ProgressBar
from models.ALBERT.model.tokenization_bert import BertTokenizer
from models.ALBERT.train_albert_pytorch.common.tools import logger
from torch.utils.data import Dataset
import pandas as pd


class NewsClassificationDataset(Dataset):
    def __init__(self,all_seq_id, all_input_ids, all_input_mask, all_segment_ids, all_label_ids):
        self.all_seq_id = all_seq_id
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_label_ids = all_label_ids

    def __getitem__(self, index):
        return tuple(attr[index] for attr in
                     (self.all_input_ids, self.all_input_mask, self.all_segment_ids, self.all_label_ids, self.all_seq_id))

    def __len__(self):
        return len(self.all_seq_id)


class InputExample(object):
    def __init__(self, guid, seq_id, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.seq_id = seq_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    '''
    A single set of features of data.
    '''

    def __init__(self,seq_id, input_ids, input_mask, segment_ids, label_id, input_len):
        self.seq_id = seq_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_len = input_len




class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, vocab_path, do_lower_case):
        self.tokenizer = BertTokenizer(vocab_path, do_lower_case)

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self, lines):
        return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1", "2"]

    @classmethod
    def read_data(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def read_data_and_create_examples(self,example_type, cached_examples_file, input_file):
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            # create examples
            df_dataset = pd.read_csv(input_file).fillna("")
            pbar = ProgressBar(n_total=len(df_dataset), desc='create examples')
            examples = []
            for i, row in df_dataset.iterrows():
                guid = '%s-%d' % (example_type, i)
                seq_id = row["id"]
                text_a = row["title"]
                text_b = row["content"]
                label = row["label"]
                label = int(label)
                example = InputExample(guid=guid, seq_id=seq_id, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
                pbar(step=i)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples


    def create_features(self, examples, max_seq_len, cached_features_file=None):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''
        pbar = ProgressBar(n_total=len(examples), desc='create features')
        if cached_features_file is not None and cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id, example in enumerate(examples):
                seq_id = example.seq_id
                tokens_a = self.tokenizer.tokenize(example.text_a)
                tokens_b = None
                label_id = example.label

                if example.text_b:
                    tokens_b = self.tokenizer.tokenize(example.text_b)
                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    self.truncate_seq_pair(tokens_a, tokens_b, max_length=max_seq_len - 3)
                else:
                    # Account for [CLS] and [SEP] with '-2'
                    if len(tokens_a) > max_seq_len - 2:
                        tokens_a = tokens_a[:max_seq_len - 2]
                tokens = ['[CLS]'] + tokens_a + ['[SEP]']
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ['[SEP]']
                    segment_ids += [1] * (len(tokens_b) + 1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [0] * (max_seq_len - len(input_ids))
                input_len = len(input_ids)

                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {example.guid}" % ())
                    logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
                    logger.info(f"label id : {label_id}")

                feature = InputFeature(seq_id=seq_id,
                                       input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       label_id=label_id,
                                       input_len=input_len)
                features.append(feature)
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)
        return features

    def create_dataset(self, features):
        # Convert to Tensors and build dataset
        all_seq_id = [f.seq_id for f in features]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        dataset = NewsClassificationDataset(all_seq_id, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
