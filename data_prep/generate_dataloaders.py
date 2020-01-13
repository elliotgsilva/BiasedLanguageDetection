#!/usr/bin/env python
# coding: utf-8

# ### Make a dictionary, dataloader

#Classes: Dictionary, TensoredDataset
#Fcns: indexize_dataset, pad_list_of_tensors, pad_collate_fn


import sys
#import jsonlines
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler, SequentialSampler
import pickle
import random 
import numpy as np

# Random seed
seed = 1029

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _init_fn(worker_id):
    np.random.seed(int(seed))


class Dictionary(object):
    def __init__(self, datasets, include_valid=False):
        self.tokens = []
        self.ids = {}
        self.counts = {}
        
        # add special tokens
        self.add_token('<pad>')
        self.add_token('<unk>')
        
        for line in tqdm(datasets['review']):
            for w in line:
                self.add_token(w)

                            
    def add_token(self, w):
        if w not in self.tokens:
            self.tokens.append(w)
            _w_id = len(self.tokens) - 1
            self.ids[w] = _w_id
            self.counts[w] = 1
        else:
            self.counts[w] += 1

    def get_id(self, w):
        return self.ids[w]
    
    def get_token(self, idx):
        return self.tokens[idx]
    
    def decode_idx_seq(self, l):
        return [self.tokens[i] for i in l]
    
    def encode_token_seq(self, l):
        return [self.ids[i] if i in self.ids else self.ids['<unk>'] for i in l]
    
    def __len__(self):
        return len(self.tokens)



def indexize_dataset(datasets, dictionary):
    indexized_datasets = []
    for l in tqdm(datasets["review"]):
        encoded_l = dictionary.encode_token_seq(l)
        indexized_datasets.append(encoded_l)
        
    return indexized_datasets



class TensoredDataset(object):
    def __init__(self, list_of_lists_of_tokens, list_of_labels,list_of_flagged_indexes):
        self.input_tensors = []
        self.label_tensors = []
        self.flagged_index = []
        self.problematic = []
        
        for sample in list_of_lists_of_tokens:
            self.input_tensors.append(torch.tensor([sample], dtype=torch.long))
        for sample in list_of_labels:
            self.label_tensors.append(torch.tensor(sample, dtype=torch.long))
        for sample in list_of_flagged_indexes:
            self.flagged_index.append(torch.tensor(sample, dtype=torch.long))
        
    def __len__(self):
        return len(self.input_tensors)
    
    def __getitem__(self, idx):
        # return a (input, target) tuple
        return (self.input_tensors[idx], self.label_tensors[idx], self.flagged_index[idx])



def pad_list_of_tensors(list_of_tensors, pad_token):
    max_length = 30
    padded_list = []
    
    for t in list_of_tensors:    
        padded_tensor = torch.cat([t, torch.tensor([[pad_token]*(max_length - t.size(-1))], dtype=torch.long)], dim = -1)
        padded_list.append(padded_tensor[:max_length])
        
    padded_tensor = torch.cat(padded_list, dim=0)
    
    return padded_tensor

def pad_collate_fn(batch):
    # batch is a list of sample tuples
    token_list = [s[0] for s in batch]
    label_list = torch.LongTensor([s[1] for s in batch])
    idx_list = torch.LongTensor([s[2] for s in batch])
    
    #pad_token = persona_dict.get_id('<pad>')
    pad_token = 0
    
    input_tensor = pad_list_of_tensors(token_list, pad_token)
    
    return input_tensor, label_list, idx_list

