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

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        '''
        Arguments:
                 x:
        '''
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos/(10000 ** ((2*i)/self.embed_dim)))
                pe[pos, i+1] = math.cos(pos/(10000 ** ((2*i)/self.embed_dim)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        def forward(self, x):
            '''
            Arguments:
                    x:
            Returns:
                    out:
            '''
            x = x*math.sqrt(self.embed_dim)

            seq_len = x.size(1)
            x = x+torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)

            return x 