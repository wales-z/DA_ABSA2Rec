import argparse
import os
import torch
import torch.nn as nn
import random
import numpy as np
import pickle

from utils import convert_examples_to_seq_features, ABSAProcessor
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from models import BertABSATagger
from dataset import get_refinetune_dataset, get_UserItem_dataset

from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler


# in this python script, we re-fine-tune the BERT-ABSA model with generated psuedo tags (related file: tagged_reviews_df.pkl)

def refinetune(args, refinetune_dataset, model, fine_tune_epochs = 3):
    print(f'doing: re-fine-tune')
    batch_size = args.per_gpu_batch_size
    refinetune_sampler = RandomSampler(refinetune_dataset)

    refinetune_dataloader = DataLoader(refinetune_dataset, sampler=refinetune_sampler, batch_size=batch_size)
    fine_tune_optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_learning_rate, eps=args.adam_epsilon)
    fine_tune_scheduler = get_linear_schedule_with_warmup(fine_tune_optimizer, num_warmup_steps=0, num_training_steps=fine_tune_epochs*len(refinetune_dataloader))

    for epoch_index in trange(fine_tune_epochs):
        fine_tune_loss = []
        for step, batch in enumerate(refinetune_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                # XLM don't use segment_ids
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                'labels': batch[3],
            }
            # tagging_loss = model(**inputs, fine_tune=True)
            (tagging_loss, tagging_logits), last_hidden_state = model(**inputs, fine_tune=True)

            fine_tune_loss.append(tagging_loss)
            tagging_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            fine_tune_optimizer.step()
            fine_tune_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
        fine_tune_average_loss = torch.mean(torch.stack(fine_tune_loss))
        print(f'epoch {epoch_index}, tagging loss: {fine_tune_average_loss}')
    print(f're-fine-tune:done')

    return model

def make_doc_embedding(args, model, user_dataset, item_dataset, tiny=False):
    print(f'doing: generate user/item doc embedding and save them to files')
    print(f'tiny: {tiny}')
    batch_size = args.per_gpu_batch_size

    print(f'doing: user part')
    user_sampler = SequentialSampler(user_dataset)
    user_dataloader = DataLoader(user_dataset, sampler=user_sampler, batch_size=batch_size)

    user_emb_dict = {}
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, tagging_labels, uids in user_dataloader:
            batch = (input_ids.to(args.device), attention_mask.to(args.device), token_type_ids.to(args.device), tagging_labels.to(args.device))
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                # XLM don't use segment_ids
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                'labels': batch[3],
            }

            (tagging_loss, tagging_logits), last_hidden_state = model(**inputs, fine_tune=False)
            for i in range(len(uids)):
                user_emb_dict[uids[i].item()] = (last_hidden_state[i].cpu(), tagging_logits[i].cpu())
    output_dir = os.path.join(args.data_dir, args.dataset, 'user_emb.pkl')
    if tiny==True:
        output_dir = os.path.join(args.data_dir, args.dataset, 'user_emb_tiny.pkl')
    with open(output_dir, 'wb') as f_u:
        pickle.dump(user_emb_dict, f_u)
    f_u.close()
    print(f'user part: done')

    print(f'doing: item part')
    item_sampler = SequentialSampler(item_dataset)
    item_dataloader = DataLoader(item_dataset, sampler=item_sampler, batch_size=batch_size)

    item_emb_dict = {}
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, tagging_labels, iids in item_dataloader:
            batch = (input_ids.to(args.device), attention_mask.to(args.device), token_type_ids.to(args.device), tagging_labels.to(args.device))
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                # XLM don't use segment_ids
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                'labels': batch[3],
            }

            (tagging_loss, tagging_logits), last_hidden_state = model(**inputs, fine_tune=False)
            for i in range(len(iids)):
                item_emb_dict[iids[i].item()] = (last_hidden_state[i].cpu(), tagging_logits[i].cpu())
    output_dir = os.path.join(args.data_dir, args.dataset, 'item_emb.pkl')
    if tiny==True:
        output_dir = os.path.join(args.data_dir, args.dataset, 'item_emb_tiny.pkl')
    with open(output_dir, 'wb') as f_i:
        pickle.dump(item_emb_dict, f_i)
    f_i.close()
    print(f'item part: done')
    print(f'generate user/item doc embedding and save them to files: done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/', type=str, required=False, 
                        help='choose the preprecess task to, choice:[preprocess, split]')
    parser.add_argument('--dataset', default='cell_phones_and_accessories', type=str, required=False, 
                        help='choose the dataset to preprocess, choice:[cell_phones_and_accessories, electronics, yelp]')
    parser.add_argument('--tiny', action='store_true', 
                        help='Weather to generate a tiny version of dataset')
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: ['bert']")
    parser.add_argument("--fine_tune_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS')
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")

    args = parser.parse_args()
    dataset_name = args.dataset
    args.task_name = dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # print(args.device)

    processor = ABSAProcessor(data_dir=args.data_dir, task_name=args.dataset, tiny=args.tiny)
    label_list = processor.get_labels(args.tagging_schema)
    num_labels = len(label_list)

    bert_config = BertConfig.from_pretrained('bert-base-uncased',
                                          num_labels=num_labels, finetuning_task=args.dataset, cache_dir="./cache")
    bert_config.absa_type = 'linear'
    bert_config.tfm_mode = 'finetune'
    bert_config.fix_tfm = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                do_lower_case=True, cache_dir='./cache')

    fine_tuned_parameter_path = os.path.join('BERT_E2E_ABSA', 'bert-tfm-laptop14-finetune', 'pytorch_model.bin')
    bertABSATagger = BertABSATagger.from_pretrained(fine_tuned_parameter_path, from_tf=False,
                                config=bert_config, cache_dir='./cache')
    bertABSATagger.to(args.device)

    refinetune_dataset, refinetune_all_evaluate_label_ids = get_refinetune_dataset(args, dataset_name, tokenizer, processor)
    refinetuned_bertABSATagger = refinetune(args, refinetune_dataset, bertABSATagger)
    refinetuned_bertABSATagger = bertABSATagger

    user_dataset, user_all_evaluate_label_ids, item_dataset, item_all_evaluate_label_ids = get_UserItem_dataset(args, dataset_name, tokenizer, processor)
    make_doc_embedding(args, refinetuned_bertABSATagger, user_dataset, item_dataset, tiny=args.tiny)

if __name__ == '__main__':
    main()
