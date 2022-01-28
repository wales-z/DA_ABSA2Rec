import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import pickle

from utils import ABSAProcessor, convert_examples_to_seq_features, convert_dict_examples_to_seq_features
from torch.utils.data import Dataset, DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

class RecDataset_ori(Dataset):
    def __init__(self, df, user_embs_dict, item_embs_dict, user_features_dict=None, item_features_dict=None):
        # self.user_embs_dict=user_embs_dict
        # self.item_embs_dict=item_embs_dict
        self.uids = torch.from_numpy(np.array(df['uid']))
        self.iids = torch.from_numpy(np.array(df['iid']))
        self.ratings = torch.from_numpy(np.array(df['overall'], dtype=np.float32))

        user_embs = sorted(user_embs_dict.items()) #词典排序后成了列表，每个元素是一个元组 (uid, (bert-embedding, tagging-logits))
        self.user_embs = torch.stack([emb[1][0] for emb in user_embs])
        self.user_tag_logis = torch.stack([emb[1][1] for emb in user_embs])

        item_embs = sorted(item_embs_dict.items())
        self.item_embs = torch.stack([emb[1][0] for emb in item_embs])
        self.item_tag_logis = torch.stack([emb[1][1] for emb in item_embs])

        user_features = sorted(user_features_dict.items())
        user_masks = torch.stack([torch.tensor(feature[1].input_mask, dtype=torch.long) for feature in user_features]).unsqueeze(dim=-1)

        item_features = sorted(item_features_dict.items())
        item_masks = torch.stack([torch.tensor(feature[1].input_mask, dtype=torch.long) for feature in item_features]).unsqueeze(dim=-1)

        self.user_embs = self.user_embs * user_masks
        self.item_embs = self.item_embs * item_masks

    def __getitem__(self, index):
        uid = self.uids[index]
        iid = self.iids[index]
        rating = self.ratings[index]

        user_emb = self.user_embs[uid]
        user_logits = self.user_tag_logis[uid]

        item_emb = self.item_embs[iid]
        item_logits = self.item_tag_logis[iid]

        return uid, user_emb, user_logits, iid, item_emb, item_logits, rating

    def __len__(self):
        return len(self.ratings)

class RecDataset(Dataset):
    def __init__(self, df):
        # self.user_embs_dict=user_embs_dict
        # self.item_embs_dict=item_embs_dict
        self.uids = torch.from_numpy(np.array(df['uid']))
        self.iids = torch.from_numpy(np.array(df['iid']))
        self.ratings = torch.from_numpy(np.array(df['overall'], dtype=np.float32))

    def __getitem__(self, index):
        uid = self.uids[index]
        iid = self.iids[index]
        rating = self.ratings[index]

        return uid, iid, rating

    def __len__(self):
        return len(self.ratings)

def get_refinetune_dataset(args, task, tokenizer, processor):
    # Load refinetune features from cache or dataset file
    mode = 'refinetune'
    cached_features_file = os.path.join(args.data_dir, task, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, 'bert-base-uncased'.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if processor.tiny == True:
        cached_features_file += '_tiny'

    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)
    else:
        # logger.info("Creating features from dataset file at %s", args.data_dir)
        print(f"Creating features from dataset file at {args.data_dir}")
        label_list = processor.get_labels(args.tagging_schema)

        examples = processor.get_refinetune_examples(args.data_dir, args.task_name, args.tagging_schema)
        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

        #logger.info("Saving features into cached file %s", cached_features_file)
        with open(cached_features_file, 'wb') as f:
            pickle.dump(features, f)
        f.close()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    # used in evaluation
    refinetune_all_evaluate_label_ids = [f.evaluate_label_ids for f in features]

    refinetune_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return refinetune_dataset, refinetune_all_evaluate_label_ids


def get_UserItem_dataset(args, task, tokenizer, processor):
    # Load user/item features from cache or dataset file
    mode = 'user'
    cached_user_features_file = os.path.join(args.data_dir, task, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, 'bert-base-uncased'.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    mode = 'item'
    cached_item_features_file = os.path.join(args.data_dir, task, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, 'bert-base-uncased'.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if processor.tiny == True:
        cached_user_features_file += '_tiny'
        cached_item_features_file += '_tiny'

    # user feature part
    if os.path.exists(cached_user_features_file):
        print("cached user features file:", cached_user_features_file)
        with open(cached_user_features_file, 'rb') as f_user:
            features = pickle.load(f_user)
    else:
        # logger.info("Creating user features from dataset file at %s", args.data_dir)
        print(f"Creating user features from dataset file at {args.data_dir}")
        label_list = processor.get_labels(args.tagging_schema)
        examples = processor.get_user_examples(args.data_dir, args.task_name, args.tagging_schema)

        features = convert_dict_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

        #logger.info("Saving features into cached file %s", cached_features_file)
        with open(cached_user_features_file, 'wb') as f_user:
            pickle.dump(features, f_user)
        f_user.close()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features.values()], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features.values()], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features.values()], dtype=torch.long)
    all_uids = torch.tensor([identifier for identifier in features.keys()], dtype=torch.int32)

    # used in evaluation
    user_all_evaluate_label_ids = [f.evaluate_label_ids for f in features.values()]

    user_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_uids)

    # item feature part
    if os.path.exists(cached_item_features_file):
        print("cached item features file:", cached_item_features_file)
        with open(cached_item_features_file, 'rb') as f_item:
            features = pickle.load(f_item)
    else:
        # logger.info("Creating item features from dataset file at %s", args.data_dir)
        print(f"Creating item features from dataset file at {args.data_dir}")
        label_list = processor.get_labels(args.tagging_schema)
        examples = processor.get_item_examples(args.data_dir, args.task_name, args.tagging_schema)

        features = convert_dict_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

        #logger.info("Saving features into cached file %s", cached_features_file)
        with open(cached_item_features_file, 'wb') as f_item:
            pickle.dump(features, f_item)
        f_item.close()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features.values()], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features.values()], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features.values()], dtype=torch.long)
    all_iids = torch.tensor([identifier for identifier in features.keys()], dtype=torch.int32)

    # used in evaluation
    item_all_evaluate_label_ids = [f.evaluate_label_ids for f in features.values()]

    item_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_iids)

    return user_dataset, user_all_evaluate_label_ids, item_dataset, item_all_evaluate_label_ids
