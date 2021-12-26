os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import os
import torch
import logging
import random
import numpy as np

from glue_utils import convert_examples_to_seq_features, output_modes, processors, compute_metrics_absa
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from transformers import AdamW, get_linear_schedule_with_warmup
from absa_layer import BertABSATagger, XLNetABSATagger

from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tensorboardX import SummaryWriter

import glob
import json

# a way to do distributed training with multiple GPUs:
'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "3, 1, 2" # this should be set before import torch

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

    args.device = device # need to set multiple GPUs

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
'''
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
    'xlnet': (XLNetConfig, XLNetABSATagger, XLNetTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--absa_type", default=None, type=str, required=True,
                        help="Downstream absa layer type selected in the list: [linear, gru, san, tfm, crf]")
    parser.add_argument("--tfm_mode", default=None, type=str, required=True,
                        help="mode of the pre-trained transformer, selected from: [finetune]")
    parser.add_argument("--fix_tfm", default=None, type=int, required=True,
                        help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
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
    args.model_type = 'bert'
    args.absa_type = 'linear'
    args.tfm_mode = 'finetune'
    args.fix_tfm = 0
    args.task_name = 'laptop14'
    args.model_name_or_path = 'bert-base-uncased'
    args.data_dir = './data/'+args.task_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize the pre-trained model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name, cache_dir="./cache")
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, cache_dir='./cache')

    config.absa_type = args.absa_type
    config.tfm_mode = args.tfm_mode
    config.fix_tfm = args.fix_tfm
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config, cache_dir='./cache')

