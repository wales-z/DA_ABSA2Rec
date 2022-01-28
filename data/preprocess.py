import json
import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import argparse
import tqdm

from collections import defaultdict
from itertools import count
from pandas.core.frame import DataFrame
from transformers import BertConfig, BertTokenizer
from sklearn.model_selection import train_test_split


dataset_to_rawfile_name = {
    'electronics': 'reviews_Electronics_5.json',
    'cell_phones_and_accessories': "reviews_Cell_Phones_and_Accessories_5.json",
    'yelp': 'yelp_academic_dataset_review.json'
}

class Preprocessor:
    def __init__(self, dataset_name, tiny=False):
        print('doing: preprocess {dataset_name} dataset')
        self.base_dir = './' + dataset_name+'/'
        rawfile_path = dataset_to_rawfile_name[dataset_name]
        raw_df = pd.read_json(rawfile_path, lines=True)

        if dataset_name == 'yelp':
            raw_df = raw_df[raw_df['cool']>0]
            raw_df = raw_df[raw_df['useful']>0]
            raw_df = raw_df[['user_id', 'business_id', 'text', 'stars']].reset_index(drop=True)
            raw_df.rename({'user_id':'reviewerID', 'business_id':'asin', 'text':'reviewText', 'stars':'overall'}, axis='columns', inplace=True)

        # 消除所有空评论,这句可能导致一个用户的交互数少于5
        df = raw_df[raw_df['reviewText']!='']

        if tiny == True:
            df = df[:30000]
            df = self.filterout(df, 5, 5).reset_index(drop=True)
            df = self.convert_idx(df)
            self.save_to_file(dataset_name, df, postfix='_tiny')
        else:
            df = self.filterout(df, 5, 5).reset_index(drop=True)
            df = self.convert_idx(df)
            self.save_to_file(dataset_name, df)
        print('preprocess {dataset_name} dataset: done')

    def filterout(self, df, thre_i, thre_u):
        index = df[["overall", "asin"]].groupby('asin').count() >= thre_i
        item = set(index[index['overall'] == True].index)
        df = df[df['asin'].isin(item)]

        index = df[["overall", "reviewerID"]].groupby(
            'reviewerID').count() >= thre_u
        user = set(index[index['overall'] == True].index)
        df = df[df['reviewerID'].isin(user)]

        return df

    def convert_idx(self, df):
        uiterator = count(0)
        udict = defaultdict(lambda: next(uiterator))
        [udict[user] for user in df["reviewerID"]]
        
        iiterator = count(0)
        idict = defaultdict(lambda: next(iiterator))
        [idict[item] for item in df["asin"]]
        
        df['uid'] = df['reviewerID'].map(lambda x: udict[x])
        df['iid'] = df['asin'].map(lambda x: idict[x])
        
        user_set = set(df["uid"])
        item_set = set(df["iid"])
        
        print(f'user num:{len(user_set)}')
        print(f'item num:{len(item_set)}')
        print(f'rating num = review num:{len(df["reviewText"])}')

        return df[["uid","iid","reviewText","overall"]]

    def get_documents(self, df, user_set, item_set):
        udocs, idocs = {}, {}
        for uid in user_set:
            string_list = df[df['uid']==uid]['reviewText'].tolist()
            udocs[uid] = ' '.join(string_list)

        for iid in item_set:
            string_list = df[df['iid']==iid]['reviewText'].tolist()
            idocs[iid] = ' '.join(string_list)

        return udocs, idocs

    def save_to_file(self, dataset_name, reviews_df, udict=None, idict=None, udocs=None, idocs=None, postfix=''):
        base_dir = "./"+dataset_name+"/"

        reviews_df['reviewText'].to_csv(base_dir+'reviews' + postfix + '.txt', index=False, header=False)
        with open(base_dir + 'reviews_df' + postfix + '.pkl', 'wb') as f_reviews_df:
            pickle.dump(reviews_df, f_reviews_df)
        f_reviews_df.close()

        if udict != None:
            with open(base_dir + 'udict' + postfix + '.pkl', 'wb') as f_udict:
                pickle.dump(udict, f_udict)
            f_udict.close()

        if idict != None:
            with open(base_dir + 'idict' + postfix + '.pkl', 'wb') as f_idict:
                pickle.dump(idict, f_idict)
            f_idict.close()

        if udocs != None:
            with open(base_dir + 'udocs' + postfix + '.pkl', 'wb') as f_udocs:
                pickle.dump(udocs, f_udocs)
            f_udocs.close()

        if idocs != None:
            with open(base_dir + 'idocs' + postfix + '.pkl', 'wb') as f_idocs:
                pickle.dump(idocs, f_idocs)
            f_idocs.close()

def trainDF_to_tagged_documents(dataset_name, df, tiny=False):
    print(f'doing: make tagged udocs/idocs from tagged reviews of train set')
    user_set = set(df["uid"])
    item_set = set(df["iid"])
    tagged_udocs, tagged_idocs = {}, {}
    for uid in user_set:
        string_list = df[df['uid']==uid]['tagged_reviews'].tolist()
        sentence_list, tag_list = [], []
        for string in string_list:
            [sentence, tags] = string.strip().split('####')
            sentence_list.append(sentence.replace('[SEP]', ''))
            tag_list.append(tags.replace('[SEP]=O', ''))
        tagged_udocs[uid] = ' '.join(sentence_list) + '####' + ' '.join(tag_list)

    for iid in item_set:
        string_list = df[df['iid']==iid]['tagged_reviews'].tolist()
        sentence_list, tag_list = [], []
        for string in string_list:
            [sentence, tags] = string.strip().split('####')
            sentence_list.append(sentence.replace('[SEP]', ''))
            tag_list.append(tags.replace('[SEP]=O', ''))
        tagged_idocs[iid] = ' '.join(sentence_list) + '####' + ' '.join(tag_list)

    if tiny == True:
        output_dir = os.path.join(dataset_name, 'tagged_udocs_tiny.pkl')
        with open(output_dir, 'wb') as f_tagged_udocs:
            pickle.dump(tagged_udocs, f_tagged_udocs)

        output_dir = os.path.join(dataset_name, 'tagged_idocs_tiny.pkl')
        with open(output_dir, 'wb') as f_tagged_idocs:
            pickle.dump(tagged_idocs, f_tagged_idocs)
    else:
        output_dir = os.path.join(dataset_name, 'tagged_udocs.pkl')
        with open(output_dir, 'wb') as f_tagged_udocs:
            pickle.dump(tagged_udocs, f_tagged_udocs)

        output_dir = os.path.join(dataset_name, 'tagged_idocs.pkl')
        with open(output_dir, 'wb') as f_tagged_idocs:
            pickle.dump(tagged_idocs, f_tagged_idocs)
    print(f'save tagged udocs and idocs: done')

def trainDF_to_documents(dataset_name, df, tiny=False):
    print(f'doing: make tagged udocs/idocs from tagged reviews of train set')
    user_set = set(df["uid"])
    item_set = set(df["iid"])
    udocs, idocs = {}, {}
    for uid in user_set:
        string_list = df[df['uid']==uid]['reviewText'].tolist()
        udocs[uid] = ' '.join(string_list)

    for iid in item_set:
        string_list = df[df['iid']==iid]['reviewText'].tolist()
        idocs[iid] = ' '.join(string_list)

    if tiny == True:
        output_dir = os.path.join(dataset_name, 'udocs_tiny.pkl')
        with open(output_dir, 'wb') as f_udocs:
            pickle.dump(udocs, f_udocs)

        output_dir = os.path.join(dataset_name, 'idocs_tiny.pkl')
        with open(output_dir, 'wb') as f_idocs:
            pickle.dump(idocs, f_idocs)
    else:
        output_dir = os.path.join(dataset_name, 'udocs.pkl')
        with open(output_dir, 'wb') as f_udocs:
            pickle.dump(udocs, f_udocs)

        output_dir = os.path.join(dataset_name, 'idocs.pkl')
        with open(output_dir, 'wb') as f_idocs:
            pickle.dump(idocs, f_idocs)
    print(f'save udocs and idocs: done')

def split_dataset(dataset_name, df, tiny = False):
    print(f'doing: split dataset')
    base_dir = "./"+dataset_name+"/reviews_df_"
    
    uid_set = set(df['uid'])
    uid_set.remove(0)
    dataset_first_user = df[df['uid']==0]
    train_dataset, test_dataset = train_test_split(dataset_first_user, test_size = 0.2)
    for uid in uid_set:
        temp_user_dataset = df[df['uid']==uid]
        temp_train_dataset, temp_test_dataset = train_test_split(temp_user_dataset, test_size = 0.2)
        train_dataset = train_dataset.append(temp_train_dataset)
        test_dataset = test_dataset.append(temp_test_dataset)

    # remove items that only appear in test set, since they have no corresponding reviews in train set thus we can't create their document
    test_dataset = test_dataset[test_dataset['iid'].isin(set(train_dataset['iid']))]

    # reset the item ids
    train_dataset['asin'] = train_dataset['iid']
    iiterator = count(0)
    idict = defaultdict(lambda: next(iiterator))
    [idict[item] for item in train_dataset["asin"]]
    train_dataset['iid'] = train_dataset['asin'].map(lambda x: idict[x])
    test_dataset['asin'] = test_dataset['iid']
    test_dataset['iid'] = test_dataset['asin'].map(lambda x: idict[x])
    max_iid = max(test_dataset['iid'])
    print(f'new item num: {max_iid+1}')
    del train_dataset['asin']
    del test_dataset['asin']

    # trainDF_to_tagged_documents(dataset_name=dataset_name, df=train_dataset, tiny=tiny)
    trainDF_to_documents(dataset_name=dataset_name, df=train_dataset, tiny=tiny)

    if tiny == True:
        with open(base_dir+'train_tiny.pkl', 'wb') as f_train_dataset:
            pickle.dump(train_dataset, f_train_dataset)
        f_train_dataset.close()

        with open(base_dir+'test_tiny.pkl', 'wb') as f_test_dataset:
            pickle.dump(test_dataset, f_test_dataset)
        f_test_dataset.close()
    else:
        with open(base_dir+'train.pkl', 'wb') as f_train_dataset:
            pickle.dump(train_dataset, f_train_dataset)
        f_train_dataset.close()

        with open(base_dir+'test.pkl', 'wb') as f_test_dataset:
            pickle.dump(test_dataset, f_test_dataset)
        f_test_dataset.close()
    print(f'split dataset: done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', default='preprocess', type=str, required=False, 
    #                     help='choose the preprecess task to, choice:[preprocess, split]')
    parser.add_argument('--dataset', default='cell_phones_and_accessories', type=str, required=False, 
                        help='choose the dataset to preprocess, choice:[cell_phones_and_accessories, electronics, yelp]')
    parser.add_argument('--tiny', action='store_true', 
                        help='Weather to generate a tiny version of dataset')


    args = parser.parse_args()
    dataset_name = args.dataset

    if not os.path.exists(dataset_name):
        os.mkdir(dataset_name)
    preprocessor = Preprocessor(dataset_name, tiny=args.tiny)

    if args.tiny == True:
        df_dir = os.path.join(dataset_name, 'reviews_df_tiny.pkl')
    else:
        df_dir = os.path.join(dataset_name, 'reviews_df.pkl')
    with open(df_dir, 'rb') as f_reviews_df:
        reviews_df = pickle.load(f_reviews_df)
    split_dataset(dataset_name, reviews_df, tiny=args.tiny)
