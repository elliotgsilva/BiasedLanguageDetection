#!/usr/bin/env python
# coding: utf-8

# ### Make a dictionary, dataloader

# In[1]:


import sys
import jsonlines
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler, SequentialSampler
import pickle


# In[2]:


# Import prepocessed Dataset(already tokenized)
with open("./data/master_df.p", 'rb') as handle:
    datasets = pickle.load(handle)


# In[3]:


datasets=datasets[datasets['review'].apply(lambda x: len(x)<=30)]


# In[4]:


class Dictionary(object):
    def __init__(self, datasets, include_valid=False):
        self.tokens = []
        self.ids = {}
        self.counts = {}
        
        # add special tokens
        self.add_token('<bos>')
        self.add_token('<eos>')
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


# In[5]:


# Make a dictionary
review_dict = Dictionary(datasets, include_valid=False)


# In[6]:


review_dict.get_id("thank")


# In[7]:


review_dict.encode_token_seq(datasets.iloc[0,0])


# In[8]:


def indexize_dataset(datasets, dictionary):
    indexized_datasets = []
    for l in tqdm(datasets["review"]):
        encoded_l = dictionary.encode_token_seq(l)
        indexized_datasets.append(encoded_l)
        
    return indexized_datasets


# In[9]:


indexized_datasets = indexize_dataset(datasets, review_dict)


# In[10]:


class TensoredDataset(object):
    def __init__(self, list_of_lists_of_tokens, list_of_flagged_indexes, list_of_problematic_flags):
        self.input_tensors = []
        self.flagged_index = []
        self.problematic = []
        
        for sample in list_of_lists_of_tokens:
            self.input_tensors.append(torch.tensor([sample[:-1]], dtype=torch.long))
        for sample in list_of_flagged_indexes:
            self.flagged_index.append(torch.tensor(sample, dtype=torch.long))
        for sample in list_of_problematic_flags:
            self.problematic.append(torch.tensor(sample, dtype=torch.long))
        
    def __len__(self):
        return len(self.input_tensors)
    
    def __getitem__(self, idx):
        # return a (input, target) tuple
        return (self.input_tensors[idx], self.flagged_index[idx], self.problematic[idx])


# In[11]:


tensor_dataset = TensoredDataset(indexized_datasets,datasets["flagged_index"].to_list(),datasets["problematic"].to_list())


# In[12]:


# check the first example
tensor_dataset[0]


# In[22]:


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
    idx_list = torch.FloatTensor([s[1] for s in batch])
    problematic = torch.FloatTensor([s[2] for s in batch])
    
    #pad_token = persona_dict.get_id('<pad>')
    pad_token = 2
    
    input_tensor = pad_list_of_tensors(token_list, pad_token)
    
    return input_tensor, idx_list, problematic


# In[23]:


# Divide into train(95%), valid(5%) dataset
batch_size = 32
n_train_samples = int(0.95 * len(datasets))
n_val_samples = len(datasets) - n_train_samples

train_dataset, val_dataset = random_split(tensor_dataset, [n_train_samples, n_val_samples])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)


# In[28]:


for i, x in enumerate(train_dataloader):
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    print(x[2])
    break


# In[15]:


path = os.getcwd()
data_dir = path + '/data/'

pickle_train_dataloader = open(data_dir + "train_dataloader.p","wb")
pickle.dump(train_dataloader, pickle_train_dataloader)
pickle_train_dataloader.close()

pickle_val_dataloader = open(data_dir + "val_dataloader.p","wb")
pickle.dump(val_dataloader, pickle_val_dataloader)
pickle_val_dataloader.close()


# In[ ]:




