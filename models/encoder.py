import numpy as np
import torch
import torch.nn as nn
from models.multiHeadAttention import MultiHeadAttention
from utils.ffn import FeedForwardNet
from utils.marks import getAttentionPadMask
from utils.positionEncoder import getSinCosEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, d_K, d_V, n_heads):
        super(EncoderLayer, self).__init__()
        self.encoderSelfAttention = MultiHeadAttention(d_model, d_K, d_V, n_heads)
        self.pos_ffn = FeedForwardNet(d_model, d_ffn)

    def forward(self, encoderInputs, encoderSelfAttentionMask):
        encoderOutputs, attention = self.encoderSelfAttention(encoderInputs,
                                                              encoderInputs,
                                                              encoderInputs,
                                                              encoderSelfAttentionMask)
        outputs = self.pos_ffn(encoderOutputs)
        return outputs, attention

class Encoder(nn.Module):
    def __init__(self, sourceVocabSize, sourceLength, d_model, d_ffn, d_K, d_V, n_heads, n_layers):
        super(Encoder, self).__init__()
        self.sourceEmbedding = nn.Embedding(sourceVocabSize, d_model)
        self.posEmbedding = nn.Embedding.from_pretrained(getSinCosEncoding(sourceLength + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ffn, d_K, d_V, n_heads) for _ in range(n_layers)])


    def forward(self, encoderInputs):
        encoderOutputs = self.sourceEmbedding(encoderInputs) \
                         + self.posEmbedding(torch.LongTensor([np.arange(encoderInputs.shape[1])]))
        # encoderSelfAttentionMask shape (batchSize, inputSeqLen, inputSeqLen)
        encoderSelfAttentionMask = getAttentionPadMask(encoderInputs, encoderInputs)
        encoderSelfAttentions = []
        for layer in self.layers:
            encoderOutputs, encoderSelfAttention = layer(encoderOutputs, encoderSelfAttentionMask)
            encoderSelfAttentions.append(encoderSelfAttention)
        return encoderOutputs, encoderSelfAttentions

