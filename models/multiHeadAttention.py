import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_K):
        super(ScaledDotProductAttention, self).__init__()
        self.d_K = d_K
    def forward(self, Q, K, V, attentionMask):
        # Q: [batch_size x n_heads x len_q x d_k]
        # K: [batch_size x n_heads x len_k x d_k]
        # V: [batch_size x n_heads x len_k x d_v]
        # scores: [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_K)
        scores.masked_fill_(attentionMask, -1e9)
        # attention: [batch_size x n_heads x len_q x len_k]
        attention = nn.Softmax(dim = -1)(scores)
        # context: [batch_size x n_heads x len_q x d_v]
        context = torch.matmul(attention, V)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_K, d_V, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_K = d_K
        self.d_V = d_V
        self.n_heads = n_heads
        # we usually require d_model = n_head * d_k
        assert self.d_model == self.n_heads * self.d_K
        self.W_Q = nn.Linear(self.d_model, self.d_K * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_K * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_V * self.n_heads)
        self.scaledDotProductAttention = ScaledDotProductAttention(self.d_K)
        self.layerNorm = nn.LayerNorm(self.d_model)
        # linear layer for final output
        self.linear = nn.Linear(self.n_heads * self.d_V, self.d_model)

    def forward(self, inputEmbeddings_Q, inputEmbeddings_K, inputEmbeddings_V, attentionMask):
        residual, batchSize = inputEmbeddings_Q, inputEmbeddings_Q.size(0)
        # map to query, key, and value space
        # Q: [batch_size x len_q x (n_heads x d_k)]
        Q = self.W_Q(inputEmbeddings_Q)
        K = self.W_K(inputEmbeddings_K)
        V = self.W_V(inputEmbeddings_V)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = Q.view(batchSize, -1, self.n_heads, self.d_K).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = K.view(batchSize, -1, self.n_heads, self.d_K).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = V.view(batchSize, -1, self.n_heads, self.d_V).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # attentionMask : [batch_size x n_heads x len_q x len_k]
        # repeat along head dimension
        attentionMask = attentionMask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attention = self.scaledDotProductAttention(q_s, k_s, v_s, attentionMask)
        # context: [batch_size x len_q x (n_heads x d_v)]
        context = context.transpose(1, 2).contiguous().view(batchSize, -1, self.n_heads * self.d_V)
        output = self.linear(context)
        # output: [batch_size x len_q x d_model]
        output = self.layerNorm(output + residual)
        return output, attention



