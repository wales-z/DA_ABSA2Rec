import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle

from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class RS_dataset(Dataset):
    """
    Generate a pytorch Dataset object with preprocessed recommender system dataset.
    """
    def __init__(self, Dataset_name):
        """
        Get dataset file input and initilize the dataset object.
        3 object will be got:
            :reviews_df (pandas.core.frame.DataFrame): dataframe include 4 columns:[uid, iid, reviewText, overall], where overall means rating(1~5)
            :udocs (dict): key is user id (uid) and value is a string list, each of the string is one review of a user.
            :idocs (dict): key is item id (iid) and value is a string list, each of the string is one review of a user.

        Args:
            Dataset_name (String): name of the used RS dataset
        """
        self.Dataset_name = Dataset_name
        bese_dir = "./data/" + Dataset_name + '/'

        with open(base_dir + 'reviews_df.pkl') as file:
            self.reviews_df = pickle.load(file)
        # with open(base_dir + 'udocs.pkl') as file:
        #     self.udocs = pickle.load(file)
        # with open(base_dir + 'udocs.pkl') as file:
        #     self.idocs = pickle.load(file)

        self.uids = np.array(self.reviews_df['uid'], dtype = np.float32)
        self.iids = np.array(self.reviews_df['iid'], dtype = np.float32)
        self.ratings = np.array(self.reviews_df['overall'], dtype = np.float32)
        file.close()

    def __getitem__(self, index):
        """
        Input the uid and iid to and return the corresponding rating and documents.

        Args:
            index (np.ndarray) : a np-array whose elements mean which rows of the review_df to sample. (batch preocessing)

        Returns: a batch of uid, iid, rating
            uid (np.ndarray): 
            iid (np.ndarray): 
            rating (np.ndarray): 
        """
        return self.uids[index], self.iids[index], self.ratings[index]

    def __len__(self):
        return len(self.reviews_df)


class ABSA_dataset(Dataset):
    """
    Generate a pytorch Dataset object with ABSA dataset.
    """
    def __init__(self):
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        pass
