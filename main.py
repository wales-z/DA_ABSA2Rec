import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import logging
import random
import numpy as np
import pickle
import time

from sklearn.model_selection import train_test_split
from utils import ABSAProcessor
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from models import BertABSATagger, DA_ABSA2Rec_new, DA_ABSA2Rec
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
    'electronics': (0, 0),
    'cell_phones_and_accessories': (27845, 10429)
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
                        help='Weather to generate a tiny version of dataset')
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

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # not using 16-bits training
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: False",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    # Set seed
    set_seed(args)

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

    # Distributed and parallel training
    # model = DA_ABSA2Rec(args, config, num_users=Dataset_configs[args.task_name][0], num_items=Dataset_configs[args.task_name][1])
    # model = DA_ABSA2Rec(args, config, num_users=27845, num_items=10429)
    model = DA_ABSA2Rec(args, num_users=1484, num_items=1840)
    # model = DA_ABSA2Rec_new(args, num_users=1484, num_items=1840)

    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    uemb_path = os.path.join(args.data_dir, args.task_name, 'user_emb.pkl')
    iemb_path = os.path.join(args.data_dir, args.task_name, 'item_emb.pkl')
    train_df_path = os.path.join(args.data_dir, args.task_name, 'tagged_reviews_df_train.pkl')
    test_df_path = os.path.join(args.data_dir, args.task_name, 'tagged_reviews_df_test.pkl')
    if args.tiny==True:
        uemb_path = os.path.join(args.data_dir, args.task_name, 'user_emb_tiny.pkl')
        iemb_path = os.path.join(args.data_dir, args.task_name, 'item_emb_tiny.pkl')
        train_df_path = os.path.join(args.data_dir, args.task_name, 'tagged_reviews_df_train_tiny.pkl')
        test_df_path = os.path.join(args.data_dir, args.task_name, 'tagged_reviews_df_test_tiny.pkl')

    with open(uemb_path, 'rb') as f2:
        user_emb_dict = pickle.load(f2)
    with open(iemb_path, 'rb') as f3:
        item_emb_dict = pickle.load(f3)

    with open(train_df_path, 'rb') as f_train:
        train_df = pickle.load(f_train)
    train_dataset = RecDataset(train_df, user_emb_dict, item_emb_dict)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=8, pin_memory=True)

    with open(test_df_path, 'rb') as f_test:
        test_df = pickle.load(f_test)
    test_dataset = RecDataset(test_df, user_emb_dict, item_emb_dict)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    if args.local_rank in [-1, 0]:
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        tb_writer = SummaryWriter(log_dir=os.path.join('log', current_time))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1*len(train_dataloader), num_training_steps=15*len(train_dataloader))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2*len(train_dataloader), gamma=0.9)

    get_mse = nn.MSELoss()
    # loss_func = nn.L1Loss()
    loss_func = nn.MSELoss()
    for epoch_index in range(int(args.num_train_epochs)):
    # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # train phase
        train_loss = []
        for batch in train_dataloader:
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            label_ratings = batch[6].to(torch.float32)
            inputs = {
                'uid':          batch[0],
                'user_emb':     batch[1],
                'user_logits':  batch[2],
                'iid':          batch[3],
                'item_emb':     batch[4],
                'item_logits':  batch[5]
            }
            predicted_ratings = model(**inputs)

            # rating_loss = 0.7*loss_func(predicted_ratings, label_ratings) + 0.3*loss_func(sentiment_ratings, label_ratings)
            rating_loss = loss_func(predicted_ratings, label_ratings)
            mse_loss = get_mse(predicted_ratings, label_ratings)
            if args.n_gpu > 1:
                rating_loss = rating_loss.mean()

            train_loss.append(mse_loss)
            rating_loss.backward()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

        train_average_mse = torch.mean(torch.stack(train_loss))
        tb_writer.add_scalar('Train/mse', train_average_mse.item(), epoch_index)
        tb_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch_index)

        # test phase
        test_loss = []
        for batch in test_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                label_ratings = batch[6].to(torch.float32)
                inputs = {
                    'uid':          batch[0],
                    'user_emb':     batch[1],
                    'user_logits':  batch[2],
                    'iid':          batch[3],
                    'item_emb':     batch[4],
                    'item_logits':  batch[5]
                }
                predicted_ratings  = model(**inputs)

            rating_loss = loss_func(predicted_ratings, label_ratings)
            mse_loss = get_mse(predicted_ratings, label_ratings)
            if args.n_gpu > 1:
                rating_loss = rating_loss.mean()

            test_loss.append(mse_loss)

        test_average_mse = torch.mean(torch.stack(test_loss))
        tb_writer.add_scalar('test/mse', test_average_mse.item(), epoch_index)
        print(f'epoch {epoch_index}, train mse: {train_average_mse}, test mse: {test_average_mse}')

    if args.local_rank in [-1, 0]:
        tb_writer.close()


if __name__ == '__main__':
    main()
