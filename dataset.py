import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import pickle

from utils import convert_examples_to_seq_features
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from utils import ABSAProcessor


def get_dataset(args, logger, task, tokenizer, mode='train', tiny=False):
    processor = ABSAProcessor(tiny=tiny)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if tiny == True:
        cached_features_file += '_tiny'
    
    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.tagging_schema)
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir, args.task_name, args.tagging_schema)
        elif mode =='eval':
            examples = processor.get_eval_examples(args.data_dir, args.task_name, args.tagging_schema)
        elif mode =='test':
            examples = processor.get_test_examples(args.data_dir, args.task_name, args.tagging_schema)
        else:
            raise Exception(f'unexpected mode: {mode}')

        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
    if args.local_rank in [-1, 0]:
        #logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_ratings = torch.tensor([f.rating for f in features], dtype=torch.float32)
    all_uids = torch.tensor([f.uid for f in features], dtype=torch.int32)
    all_iids = torch.tensor([f.iid for f in features], dtype=torch.int32)

    # used in evaluation
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ratings, all_uids, all_iids)
    return dataset, all_evaluate_label_ids
