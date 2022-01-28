# from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import pickle
from io import open

from seq_utils import *


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class SeqInputFeatures(object):
    """A single set of features of data for the ABSA task"""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids=None, evaluate_label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # mapping between word index and head token index
        self.evaluate_label_ids = evaluate_label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines



logger = logging.getLogger(__name__)


class ABSAProcessor(DataProcessor):
    def __init__(self, data_dir, task_name, tiny=False):
        self.tiny = tiny
        if self.tiny == True:
            self.postfix = '_tiny'
        else:
            self.postfix = ''

        # re-fine-tune 只使用训练集的 review 和 psuedo tags
        train_df_path = os.path.join(data_dir, task_name, 'reviews_df_train' + self.postfix+ '.pkl')
        with open(train_df_path, 'rb') as f_train_df:
            self.train_df = pickle.load(f_train_df)
        f_train_df.close()

        udocs_path = os.path.join(data_dir, task_name, 'udocs' + self.postfix+ '.pkl')
        idocs_path = os.path.join(data_dir, task_name, 'idocs' + self.postfix+ '.pkl')

        with open(udocs_path, 'rb') as f_udocs:
            self.udocs = pickle.load(f_udocs)

        with open(idocs_path, 'rb') as f_idocs:
            self.idocs = pickle.load(f_idocs)

    """Processor for the ABSA datasets"""
    def get_refinetune_examples(self, data_dir, task_name, tagging_schema):
        return self._create_examples(data_dir=data_dir, task_name=task_name, mode='refinetune', tagging_schema=tagging_schema, df=self.train_df)

    def get_user_examples(self, data_dir, task_name, tagging_schema):
        return self._create_dict_examples(data_dir=data_dir, task_name=task_name, mode='user', tagging_schema=tagging_schema, docs=self.udocs)

    def get_item_examples(self, data_dir, task_name, tagging_schema):
        return self._create_dict_examples(data_dir=data_dir, task_name=task_name, mode='item', tagging_schema=tagging_schema, docs=self.idocs)

    # def get_train_examples(self, data_dir, task_name, tagging_schema):
    #     return self._create_examples(data_dir=data_dir, task_name=task_name, mode='train', tagging_schema=tagging_schema)

    # def get_dev_examples(self, data_dir, task_name, tagging_schema):
    #     return self._create_examples(data_dir=data_dir, task_name=task_name, mode='dev', tagging_schema=tagging_schema)

    # def get_test_examples(self, data_dir, task_name, tagging_schema):
    #     return self._create_examples(data_dir=data_dir, task_name=task_name, mode='test', tagging_schema=tagging_schema)

    def get_labels(self, tagging_schema):
        if tagging_schema == 'OT':
            return []
        elif tagging_schema == 'BIO':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'BIEOS':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
            'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
            'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)

    def _create_dict_examples(self, data_dir, task_name, tagging_schema, docs, mode='user'):
        def string_to_word_list(s):
            s = s.replace(":",'')
            s = s.replace(",",'')
            s = s.replace(".",'')
            s = s.replace("!",'')
            s = s.replace("?",'')
            s = s.replace('\"', '')
            s = s.strip()
            return s.split()

        examples = {}
        sample_id = 0
        for identifier, line in docs.items():
            sent_string = line.strip()
            words = string_to_word_list(sent_string)

            guid = "%s-%s" % (mode, sample_id)
            text_a = ' '.join(words)
            
            examples[identifier] = InputExample(guid=guid, text_a=text_a, text_b=None, label=None)
            sample_id += 1
        return examples

    def _create_examples(self, data_dir, task_name, tagging_schema, df, mode='refinetune'):
        examples = []
        sample_id = 0
        for line in df['tagged_reviews'].tolist():
            sent_string, tag_string = line.strip().split('####')
            words = []
            tags = []
            for tag_item in tag_string.split(' '):
                eles = tag_item.split('=')
                if len(eles) == 1:
                    raise Exception("Invalid samples %s..." % tag_string)
                elif len(eles) == 2:
                    word, tag = eles
                else:
                    word = ''.join((len(eles) - 2) * ['='])
                    tag = eles[-1]
                words.append(word)
                tags.append(tag)
            # convert from ot to bieos
            if tagging_schema == 'BIEOS':
                tags = ot2bieos_ts(tags)
            elif tagging_schema == 'BIO':
                tags = ot2bio_ts(tags)
            else:
                # original tags follow the OT tagging schema, do nothing
                pass
            guid = "%s-%s" % (mode, sample_id)
            text_a = ' '.join(words)
            #label = [absa_label_vocab[tag] for tag in tags]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags))
            sample_id += 1
        return examples

def convert_examples_to_seq_features(examples, label_list, tokenizer,
                                     cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]',
                                     sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0,
                                     sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0,
                                     mask_padding_with_zero=True):
    # feature extraction for sequence labeling
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_seq_length = -1
    examples_tokenized = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = []
        labels_a = []
        evaluate_label_ids = []
        words = example.text_a.split(' ')
        wid, tid = 0, 0
        for word, label in zip(words, example.label):
            subwords = tokenizer.tokenize(word)
            tokens_a.extend(subwords)
            if label != 'O':
                labels_a.extend([label] + ['EQ'] * (len(subwords) - 1))
            else:
                labels_a.extend(['O'] * len(subwords))
            evaluate_label_ids.append(tid)
            wid += 1
            # move the token pointer
            tid += len(subwords)
        #print(evaluate_label_ids)
        assert tid == len(tokens_a)
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        examples_tokenized.append((tokens_a, labels_a, evaluate_label_ids))
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    max_seq_length = min(510, max_seq_length)
    # count on the [CLS] and [SEP]
    max_seq_length += 2
    #max_seq_length = 128
    for ex_index, (tokens_a, labels_a, evaluate_label_ids) in enumerate(examples_tokenized):
        #tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        # for sequence labeling, better not truncate the sequence
        #if len(tokens_a) > max_seq_length - 2:
        #    tokens_a = tokens_a[:(max_seq_length - 2)]
        #    labels_a = labels_a
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        labels = labels_a + ['O']
        if cls_token_at_end:
            # evaluate label ids not change
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            labels = labels + ['O']
        else:
            # right shift 1 for evaluate label ids
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            labels = ['O'] + labels
            evaluate_label_ids += 1
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        #print("Current labels:", labels)
        label_ids = [label_map[label] for label in labels]

        # pad the input sequence and the mask sequence
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # pad sequence tag 'O'
            label_ids = ([0] * padding_length) + label_ids
            # right shift padding_length for evaluate_label_ids
            evaluate_label_ids += padding_length
        else:
            # evaluate ids not change
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            # pad sequence tag 'O'
            label_ids = label_ids + ([0] * padding_length)

        # 对长序列进行截断处理
        input_ids = input_ids[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        label_ids = label_ids[:max_seq_length]
        # print(f'len(input_ids)={len(input_ids)}, len(input_mask)={len(input_mask)}, len(segment_ids)={len(segment_ids)}')

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            SeqInputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_ids=label_ids,
                             evaluate_label_ids=evaluate_label_ids
                             ))
    print("maximal sequence length is", max_seq_length)
    return features

def convert_dict_examples_to_seq_features(examples, label_list, tokenizer,
                                     cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]',
                                     sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0,
                                     sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0,
                                     mask_padding_with_zero=True):
    # 做了点改动，输入的 examples 和 返回的 features 都变成了词典类型，键是 uid/iid，值是原来的 example 或 feature 对象
    # feature extraction for sequence labeling
    label_map = {label: i for i, label in enumerate(label_list)}
    features = {}
    max_seq_length = -1
    examples_tokenized = {}
    for identifier, example in examples.items():
        tokens_a = []
        evaluate_label_ids = []
        words = example.text_a.split(' ')
        wid, tid = 0, 0
        for word in words:
            subwords = tokenizer.tokenize(word)
            tokens_a.extend(subwords)
            evaluate_label_ids.append(tid)
            wid += 1
            # move the token pointer
            tid += len(subwords)

        #print(evaluate_label_ids)
        assert tid == len(tokens_a)
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        examples_tokenized[identifier] = (tokens_a, evaluate_label_ids)
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    max_seq_length = min(510, max_seq_length)
    # count on the [CLS] and [SEP]
    max_seq_length += 2
    #max_seq_length = 128
    for identifier, (tokens_a, evaluate_label_ids) in examples_tokenized.items():
        #tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        # for sequence labeling, better not truncate the sequence
        #if len(tokens_a) > max_seq_length - 2:
        #    tokens_a = tokens_a[:(max_seq_length - 2)]
        #    labels_a = labels_a
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            # evaluate label ids not change
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            # right shift 1 for evaluate label ids
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            evaluate_label_ids += 1
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        # pad the input sequence and the mask sequence
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # right shift padding_length for evaluate_label_ids
            evaluate_label_ids += padding_length
        else:
            # evaluate ids not change
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        # 对长序列进行截断处理
        input_ids = input_ids[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        # print(f'len(input_ids)={len(input_ids)}, len(input_mask)={len(input_mask)}, len(segment_ids)={len(segment_ids)}')

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features[identifier] = SeqInputFeatures(input_ids=input_ids,
                                                input_mask=input_mask,
                                                segment_ids=segment_ids,
                                                evaluate_label_ids=evaluate_label_ids
                                                )
    print("maximal sequence length is", max_seq_length)
    return features

processors = {
    "laptop14": ABSAProcessor,
    "rest_total": ABSAProcessor,
    "rest_total_revised": ABSAProcessor,
    "rest14": ABSAProcessor,
    "rest15": ABSAProcessor,
    "rest16": ABSAProcessor,
    "electronics": ABSAProcessor,
    "cell_phones_and_accessories": ABSAProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "laptop14": "classification",
    "rest_total": "classification",
    "rest14": "classification",
    "rest15": "classification",
    "rest16": "classification",
    "rest_total_revised": "classification",
    "electronics": "classification",
    "cell_phones_and_accessories": "classification",
}
