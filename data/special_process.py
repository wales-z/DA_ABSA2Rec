import pandas as pd
import os
import pickle

# note here! before using this, you should rename the preprocessed file: reviews_df.pkl to reviews_df_ori.pkl

### electronics ###
# index_list = [0, 211000, 422000, 633000, 844000, 1055000, 1266000, 1477000, 1687366]
### electronics ###

### yelp ###
# index_list = [0, 211000, 422000, 633000, 844000, 1005090]
### yelp ###

# choose electronics or yelp
dataset_name = 'electronics'
df_ori_path = os.path.join(dataset_name, 'reviews_df_ori.pkl')


with open(df_ori_path, 'rb') as f:
    df_ori = pickle.load(f)

# choose the correct index for the correspoding chunk
df = df_ori

output_txt_dir = os.path.join(dataset_name, 'reviews.txt')
df['reviewText'].to_csv(output_txt_dir, index=False, header=False)

# and after generating psuedo tags for each chunk, remember to rename it acoording to the chunk id.
# finally you should gather all chunks into a full df with tagged_reviews, file name: tagged_reviews_df.pkl