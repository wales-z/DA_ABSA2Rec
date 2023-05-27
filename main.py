import argparse
import os
import torch
import torch.nn as nn
import logging
import random
import numpy as np
import pickle
import time
import gc
import copy

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import ABSAProcessor
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from models import BertABSATagger, DA_ABSA2Rec_new, DA_ABSA2Rec, ContrastiveLoss
from dataset import RecDataset

from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tensorboardX import SummaryWriter


# print(torch.__version__) # 1.10.0+cu102
# print(torch.version.cuda) # 10.2
# print(torch.backends.cudnn.version()) # 7605
# print(torch.cuda.get_device_name(0)) # GeForce RTX 2080 Ti

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


logger = logging.getLogger(__name__)

ALL_MODELS = (
 'bert-base-uncased',
 'bert-large-uncased',
 'bert-base-cased',
 'bert-large-cased',
 'bert-base-multilingual-uncased',
 'bert-base-multilingual-cased',
 'bert-base-chinese',
 'bert-base-german-cased',
 'bert-large-uncased-whole-word-masking',
 'bert-large-cased-whole-word-masking',
 'bert-large-uncased-whole-word-masking-finetuned-squad',
 'bert-large-cased-whole-word-masking-finetuned-squad',
 'bert-base-cased-finetuned-mrpc',
 'bert-base-german-dbmdz-cased',
 'bert-base-german-dbmdz-uncased',
 'xlnet-base-cased',
 'xlnet-large-cased'
)

MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
}

# format: (num_users, num_items)
Dataset_configs = {
    'electronics': (192238, 62973),
    'cell_phones_and_accessories': (27837, 10419),
    'yelp': (50059, 61634),
    'video_games': (24293, 10668),
    'automotive': (2923, 1833),
    'musical_instruments': (1425, 900)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--absa_type", default='linear', type=str,
                        help="Downstream absa layer type selected in the list: [linear, gru, san, tfm, crf]")
    parser.add_argument("--tfm_mode", default='finetune', type=str,
                        help="mode of the pre-trained transformer, selected from: [finetune]")
    parser.add_argument("--fix_tfm", default=0, type=int,
                        help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, 
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default='cell_phones_and_accessories', type=str,
                        help="The name of the task to train selected in the list:[]")
    parser.add_argument('--tiny', action='store_true', 
                        help='Weather to use a tiny version of dataset')
    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--scheduler_gamma", default=0.9, type=float,
                        help="the multiplier for learning rate scheduler")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
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
    parser.add_argument('--tagging_schema', type=str, default='BIEOS')

    parser.add_argument("--overfit", type=int, default=0, help="if evaluate overfit or not")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    args = parser.parse_args()
    output_dir = '%s-%s-%s-%s' % (args.model_type, args.absa_type, args.task_name, args.tfm_mode)

    if args.fix_tfm:
        output_dir = '%s-fix' % output_dir
    if args.overfit:
        output_dir = '%s-overfit' % output_dir
        args.max_steps = 3000
    args.output_dir = output_dir
    return args


def main():
    args = init_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_ADDR'] = args.MASTER_ADDR
        os.environ['MASTER_PORT'] = args.MASTER_PORT
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=1)
        args.n_gpu = 1

    args.device = device
    # args.device = torch.device('cpu')

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # not using 16-bits training
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    # Set seed
    set_seed(args)

    print(f'Now doing: load Rec model')
    # initialize the pre-trained model
    processor = ABSAProcessor(data_dir=args.data_dir, task_name=args.task_name, tiny=args.tiny)
    label_list = processor.get_labels(args.tagging_schema)
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = BertConfig, BertABSATagger, BertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name, cache_dir="./cache")
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, cache_dir='./cache')

    config.absa_type = args.absa_type
    config.tfm_mode = args.tfm_mode
    config.fix_tfm = args.fix_tfm

    num_users = Dataset_configs[args.task_name][0]
    num_items = Dataset_configs[args.task_name][1]
    print(f'dataset: {args.task_name}, user num: {num_users}, item num: {num_items}')

    # Distributed and parallel training
    model = DA_ABSA2Rec(args, num_users=num_users, num_items=num_items)
    # model = DA_ABSA2Rec(args, num_users=27845, num_items=10429)
    # model = DA_ABSA2Rec(args, num_users=1484, num_items=1840)
    # model = DA_ABSA2Rec_new(args, num_users=1484, num_items=1840)

    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    print(f'Done: load Rec model')

    print(f'Now doing: load user/item embs and features, load train/test set')
    uemb_path = os.path.join(args.data_dir, args.task_name, 'user_emb.pkl')
    iemb_path = os.path.join(args.data_dir, args.task_name, 'item_emb.pkl')
    ufeature_path = os.path.join(args.data_dir, args.task_name, 'cached_user_bert-base-uncased_512_'+args.task_name)
    ifeature_path = os.path.join(args.data_dir, args.task_name, 'cached_item_bert-base-uncased_512_'+args.task_name)
    train_df_path = os.path.join(args.data_dir, args.task_name, 'reviews_df_train.pkl')
    test_df_path = os.path.join(args.data_dir, args.task_name, 'reviews_df_test.pkl')
    if args.tiny==True:
        uemb_path = os.path.join(args.data_dir, args.task_name, 'user_emb_tiny.pkl')
        iemb_path = os.path.join(args.data_dir, args.task_name, 'item_emb_tiny.pkl')
        ufeature_path = os.path.join(args.data_dir, args.task_name, 'cached_user_bert-base-uncased_512_'+args.task_name+'_tiny')
        ifeature_path = os.path.join(args.data_dir, args.task_name, 'cached_item_bert-base-uncased_512_'+args.task_name+'_tiny')
        train_df_path = os.path.join(args.data_dir, args.task_name, 'reviews_df_train_tiny.pkl')
        test_df_path = os.path.join(args.data_dir, args.task_name, 'reviews_df_test_tiny.pkl')

    with open(uemb_path, 'rb') as f2:
        user_embs_dict = pickle.load(f2)
    with open(iemb_path, 'rb') as f3:
        item_embs_dict = pickle.load(f3)
    with open(ufeature_path, 'rb') as f4:
        user_features_dict = pickle.load(f4)
    with open(ifeature_path, 'rb') as f5:
        item_features_dict = pickle.load(f5)

    user_embs_list = sorted(user_embs_dict.items()) #词典排序后成了列表，每个元素是一个元组 (uid, (bert-embedding, tagging-logits))
    user_embs = torch.stack([emb[1][0] for emb in user_embs_list])
    user_tag_logis = torch.stack([emb[1][1] for emb in user_embs_list])
    user_features = sorted(user_features_dict.items())
    user_masks = torch.stack([torch.tensor(feature[1].input_mask, dtype=torch.long) for feature in user_features]).unsqueeze(dim=-1)
    # user_embs = user_embs * user_masks

    item_embs_list = sorted(item_embs_dict.items())
    item_embs = torch.stack([emb[1][0] for emb in item_embs_list])
    item_tag_logis = torch.stack([emb[1][1] for emb in item_embs_list])
    item_features = sorted(item_features_dict.items())
    item_masks = torch.stack([torch.tensor(feature[1].input_mask, dtype=torch.long) for feature in item_features]).unsqueeze(dim=-1)
    # item_embs = item_embs * item_masks

    print(f'Done: load user/item embs and features, load train/test set')

    with open(train_df_path, 'rb') as f_train:
        train_df = pickle.load(f_train)
    # train_dataset = RecDataset(train_df, user_emb_dict, item_emb_dict, user_features_dict, item_features_dict)
    train_dataset = RecDataset(train_df)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    with open(test_df_path, 'rb') as f_test:
        test_df = pickle.load(f_test)
    # test_dataset = RecDataset(test_df, user_emb_dict, item_emb_dict, user_features_dict, item_features_dict)
    test_dataset = RecDataset(test_df)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # Conduct garbage collection to save memory usage
    gc.collect()

    if args.local_rank in [-1, 0]:
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        tb_writer = SummaryWriter(log_dir=os.path.join('tb_log', current_time))

    param = {}
    param_s = {}
    for n, p in model.named_parameters():
        if 'concat_to_1_s' in n or 'classifier' in n:
            param_s[n]=p
        else:
            param[n]=p
    # print(param_s)

    optimizer = torch.optim.AdamW(param.values(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    optimizer_s = torch.optim.AdamW(param_s.values(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1*len(train_dataloader), num_training_steps=15*len(train_dataloader))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1*len(train_dataloader), gamma=0.9)
    scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=1*len(train_dataloader), gamma=0.9)

    # loss_func = nn.L1Loss()
    loss_func = nn.MSELoss()
    # loss_func = nn.CrossEntropyLoss()
    # loss_func = ContrastiveLoss(margin=4)

    save_prefix = os.path.join('saved_model', args.task_name)
    if not os.path.exists(save_prefix):
        os.mkdir(save_prefix)
    model_save_path = os.path.join(save_prefix, 'model.pth')

    print(f'start training, batch size: {args.per_gpu_train_batch_size}, learning rate: {args.learning_rate}')

    train_mse_recorder, test_mse_recorder = [], []
    min_mse = 5

    for epoch_index in range(int(args.num_train_epochs)):
    # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # train phase
        all_rating_pred = []
        all_rating_true = []
        total_loss = 0
        # for step, (uid, iid, label_ratings)  in enumerate(tqdm(train_dataloader, desc='training')):
        for step, (uid, iid, label_ratings)  in enumerate(train_dataloader):
            all_rating_true.extend(label_ratings.numpy())

            user_emb = user_embs[uid].to(args.device)
            user_mask = user_masks[uid].to(args.device)
            user_emb = user_emb * user_mask
            user_logits = user_tag_logis[uid].to(args.device)

            item_emb = item_embs[iid].to(args.device)
            item_mask = item_masks[iid].to(args.device)
            item_emb = item_emb * item_mask
            item_logits = item_tag_logis[iid].to(args.device)

            # label_ratings = label_ratings-1
            uid, iid, label_ratings = uid.to(args.device), iid.to(args.device), label_ratings.to(args.device)

            model.train()
            # label_ratings = batch[2].to(torch.float32)
            inputs = {
                'uid':          uid,
                'user_emb':     user_emb,
                'user_logits':  user_logits,
                'iid':          iid,
                'item_emb':     item_emb,
                'item_logits':  item_logits
            }
            # predicted_ratings_logits, sentiment_ratings_logits, u, i = model(**inputs)
            predicted_ratings, sentiment_ratings, u, i = model(**inputs)
            # predicted_ratings = torch.argmax(predicted_ratings_logits, dim=-1)+1
            # sentiment_ratings = torch.argmax(sentiment_ratings_logits, dim=-1)+1
            if predicted_ratings.dim() == 0:
                all_rating_pred.append(predicted_ratings.data)
            else:
                all_rating_pred.extend(predicted_ratings.data)
            # label_ratings = label_ratings.to(torch.long)
            # 系数范围 [0.1, 1, 10]
            # rating_loss = loss_func(predicted_ratings_logits, sentiment_ratings_logits, label_ratings) + args.weight_decay * torch.linalg.vector_norm(u) + args.weight_decay * torch.linalg.vector_norm(i)
            rating_loss = loss_func(predicted_ratings, label_ratings) + 8*loss_func(sentiment_ratings, predicted_ratings) + args.weight_decay * torch.linalg.vector_norm(u) + args.weight_decay * torch.linalg.vector_norm(i)
            rating_loss = rating_loss.requires_grad()

            if args.n_gpu > 1:
                rating_loss = rating_loss.mean()

            total_loss += rating_loss
            rating_loss.backward()

            if step%2==0 :
                optimizer.step()
                # scheduler.step()
            else:
                optimizer_s.step()
                # scheduler_s.step()
              # Update learning rate schedule
            model.zero_grad()

        print(f'loss:{total_loss}')
        train_average_mse = mean_squared_error(np.array(all_rating_true), np.array(all_rating_pred))
        tb_writer.add_scalar('Train/mse', train_average_mse, epoch_index)
        tb_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch_index)

        # test phase
        all_rating_pred = []
        all_rating_true = []
        model.eval()
        for (uid, iid, label_ratings) in test_dataloader:
            all_rating_true.extend(label_ratings.numpy())

            user_emb = user_embs[uid].to(args.device)
            user_mask = user_masks[uid].to(args.device)
            user_emb = user_emb * user_mask
            user_logits = user_tag_logis[uid].to(args.device)

            item_emb = item_embs[iid].to(args.device)
            item_mask = item_masks[iid].to(args.device)
            item_emb = item_emb * item_mask
            item_logits = item_tag_logis[iid].to(args.device)

            label_ratings = label_ratings-1
            uid, iid, label_ratings = uid.to(args.device), iid.to(args.device), label_ratings.to(args.device)

            with torch.no_grad():
                # label_ratings = batch[6].to(torch.float32)
                inputs = {
                    'uid':          uid,
                    'user_emb':     user_emb,
                    'user_logits':  user_logits,
                    'iid':          iid,
                    'item_emb':     item_emb,
                    'item_logits':  item_logits
                }
                predicted_ratings_logits, sentiment_ratings, u, i = model(**inputs)
                predicted_ratings = torch.argmax(predicted_ratings_logits, dim=-1)+1
                if predicted_ratings.dim() == 0:
                    all_rating_pred.append(predicted_ratings.data)
                else:
                    all_rating_pred.extend(predicted_ratings.data)

            label_ratings = label_ratings.to(torch.long)
            # rating_loss = loss_func(predicted_ratings_logits, label_ratings)
            # if args.n_gpu > 1:
            #     rating_loss = rating_loss.mean()

        test_average_mse = mean_squared_error(np.array(all_rating_true), np.array(all_rating_pred))
        test_average_mae = mean_absolute_error(np.array(all_rating_true), np.array(all_rating_pred))
        tb_writer.add_scalar('test/mse', test_average_mse, epoch_index)
        print(f'epoch {epoch_index}, train mse: {train_average_mse}, test mse: {test_average_mse}, test mae: {test_average_mae}')
        train_mse_recorder.append(train_average_mse)
        test_mse_recorder.append(test_average_mse)
        if(test_average_mse < min_mse):
            min_mse = test_average_mse
            best_model_state = copy.deepcopy(model.state_dict()) 
            torch.save(best_model_state, model_save_path)

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    
    print(f'train mse records: {train_mse_recorder}')
    print(f'test mse records: {test_mse_recorder}')
    print(f'finish train&eval, best performance: test mse: {min_mse}')


if __name__ == '__main__':
    main()
