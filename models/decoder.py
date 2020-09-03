import torch
import torch.nn as nn
import numpy as np
from models.multiHeadAttention import MultiHeadAttention
from utils.ffn import FeedForwardNet
from utils.positionEncoder import getSinCosEncoding
from utils.marks import getAttentionPadMask, getAttentionSubsequentMask

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, d_K, d_V, n_heads):
        super(DecoderLayer, self).__init__()
        # each decoder layer has two types of multi-head attentions
        self.decoderSelfAttention = MultiHeadAttention(d_model, d_K, d_V, n_heads)
        self.decoderEncoderAttention = MultiHeadAttention(d_model, d_K, d_V, n_heads)
        self.pos_ffn = FeedForwardNet(d_model, d_ffn)

    def forward(self, decoderInputs, encoderOutputs, decoderSelfAttentionMask, decoderEncoderAttentionMask):
        decoderOutputs, decoderSelfAttention = self.decoderSelfAttention(decoderInputs,
                                                                         decoderInputs,
                                                                         decoderInputs,
                                                                         decoderSelfAttentionMask)
        decoderOutputs, decoderEncoderAttention = self.decoderEncoderAttention(decoderOutputs,
                                                                               encoderOutputs,
                                                                               encoderOutputs,
                                                                               decoderEncoderAttentionMask)
        decoderOutputs = self.pos_ffn(decoderOutputs)

        return decoderOutputs, decoderSelfAttention, decoderEncoderAttention

class Decoder(nn.Module):
    def __init__(self,  targetVocabSize, targetLength, d_model, d_ff, d_K, d_V, n_heads, n_layers):
        super(Decoder, self).__init__()
        self.targetEmbedding = nn.Embedding(targetVocabSize, d_model)
        self.posEmbedding = nn.Embedding.from_pretrained(getSinCosEncoding(targetLength + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_K, d_V, n_heads) for _ in range(n_layers)])

    def forward(self, decoderInputs, encoderInputs, encoderOutputs):
        decoderOutputs = self.targetEmbedding(decoderInputs) \
                         + self.posEmbedding(torch.LongTensor([np.arange(decoderInputs.shape[1])]))
        decoderSelfAttentionPadMask = getAttentionPadMask(decoderInputs, decoderInputs)
        decoderSelfAttentionSubsequentMask = getAttentionSubsequentMask(decoderInputs)
        # combine the two masks via logical or for self-attention mask
        decoderSelfAttentionMask = decoderSelfAttentionPadMask | decoderSelfAttentionSubsequentMask

        decoderEncoderAttentionMask = getAttentionPadMask(decoderInputs, encoderInputs)

        decoderSelfAttentions, decoderEncoderAttentions = [], []

        for layer in self.layers:
            decoderOutputs, decoderSelfAttention, decoderEnconderAttention = layer(decoderOutputs,
                                                                                   encoderOutputs,
                                                                                   decoderSelfAttentionMask,
                                                                                   decoderEncoderAttentionMask)
            decoderSelfAttentions.append(decoderSelfAttention)
            decoderEncoderAttentions.append(decoderEnconderAttention)

        return decoderOutputs, decoderSelfAttentions, decoderEncoderAttentions

