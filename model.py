#!/usr/bin/env python
# coding: utf-8

# ### Define models

#Classes: neuralNetBow
#Fcns: load_model

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle as pkl
import os
import datetime as dt
import pandas as pd
import random 
import numpy as np
from tqdm import tqdm_notebook as tqdm

# Baseline model
class neuralNetBow(nn.Module):
    """
    BagOfWords classification model
    """
    # NOTE: we can't use linear layer until we take weighted average, otherwise it will
    # remember certain positions incorrectly (ie, 4th word has bigger weights vs 7th word)
    def __init__(self, vocab_size, emb_dim, upweight=10):
        super(neuralNetBow, self).__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=2)
        self.upweight = upweight
    
    def forward(self, tokens, flagged_index):
        batch_size, num_tokens = tokens.shape
        embedding = self.embed(tokens)
        embedding[torch.LongTensor(range(batch_size)),flagged_index.type(torch.LongTensor),:] *= self.upweight
        
        # average across embeddings
        embedding_ave = embedding.sum(1) / (num_tokens + self.upweight - 1)
        
        return embedding_ave