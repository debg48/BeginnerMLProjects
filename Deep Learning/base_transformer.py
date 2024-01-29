import torch
import torch.nn as nn
import torch.nn.functional as F 

import pandas as pd 
import numpy as np 
import seaborn as sns 

import math 
import copy 
import re 

import warnings
import torchtext 

#Creating Word Embeddings 

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        '''
        Arguments:
                 vocab_size:
                 embed_dim:
        '''
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        '''
        Arguments:
                 x:
        Returns:
                 out:
        '''
        out = self.embed(x)
        return out