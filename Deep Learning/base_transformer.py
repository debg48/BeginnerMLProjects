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

class TransfomerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor = 4, n_heads = 8):
        super(TransfomerBlock, self).__init()
        '''
        Arguments:
                 embed_dim:
                 expansion_factor:
                 n_heads:
        '''
        self.attention = MultiheadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.sequential(nn.Linear(embed_dim, expansion_factor*embed_dim),
                                          nn.ReLU(),
                                          nn.Linear(expansion_factor*embed_dim)
                                          )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        '''
        Arguments:
                 key:
                 query:
                 value:
                 norm2_out:
        '''
        attention_out = self.attention(key, query, value)

        attention_residual_out = attention_out + value 

        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)

        feed_fwd_residual_out = feed_fwd_out + norm1_out 

        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out 

class TransformerEncoder(nn.Module):
    '''
    Arguments:
             seq_len:
             embed_dim:
             num_layers:
             expansion_factors:
             n_heads:

    Returns:
             out:
    '''

    def __init__(self, seq_len, vocab_size, embed_dim, num_layers = 2, expansion_factor = 4, n_heads = 8):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)

        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range (num_layers)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)

        out = self.positional_encoder(embed_out)

        for layer in self.layers:
            out = layer(out, out, out)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor = 4, n_heads = 8):
        super(DecoderBlock, self).__init__()
        '''
        Arguments:
                 embed_dim:
                 expansion_factor:
                 n_heads:  
        '''
        self.attention = MultiheadAttention(embed_dim, n_heads = 8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransfomerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        '''
        Arguments:
                 key:
                 query:
                 value:
                 mask:
        Returns:
                 out:  
        '''
        attention = self.attention(x, x, x, mask=mask)
        value = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, query, value)

        return out 
