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

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim = 512, n_heads = 8):
        '''
        Arguments:
                 embed_dim:

                 n_heads:
        '''
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim 
        self.n_heads = n_heads 
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim, self_single_head_dim, bias = False)

        self.key_matrix = nn.Linear(self.single_head_dim, self_single_head_dim, bias = False)

        self.valur_matrix = nn.Linear(self.single_head_dim, self_single_head_dim, bias = False)

        self.out = nn.Linear(self.n_heads*self.single_headed_dim, self.embed_dim)

    def forward(self, key, query, value, mask = None):
        '''
        Arguments:
                 key:
                 query:
                 value:
                 mask:
        '''
        batch_size = key.size(0)
        seq_length = key.size(1)

        seq_length_query = query.size(1)

        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)

        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_adjusted = k.transpose(-1, -2)

        product = torch.matmul(q, k_adjusted)

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

            product = product / math.sqrt(self.single_head_dim)

            scores = F.softmax(product, dim = -1)

            scores = torch.matmul(scores, v)

            concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)

            output = self.out(concat)

            return output 