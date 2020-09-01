import torch
import torch.nn as nn
from models.multiHeadAttention import MultiHeadAttention
from utils.ffn import FeedForwardNet
from utils.marks import getAttentionPadMask
from utils.positionEncoder import getSinCosEncoding
from models.encoder import EncoderLayer

class BERT(nn.Module):
    def __init__(self, vocabSize, seqLength, d_model, n_hidden, d_K, d_V,  n_layers, n_heads):
        super(BERT, self).__init__()

        d_ffn = n_hidden * 4

        self.sourceEmbedding = nn.Embedding(vocabSize, d_model)
        self.posEmbedding = nn.Embedding.from_pretrained(getSinCosEncoding(seqLength + 1, d_model), freeze=True)
        # segment info embedding
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        self.segmentEmbedding = nn.Embedding(2, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ffn, d_K, d_V, n_heads) for _ in range(n_layers)])

    def forward(self, encoderInputs, segmentInfo):
        encoderOutputs = self.sourceEmbedding(encoderInputs) \
                         + self.posEmbedding(torch.LongTensor([[1, 2, 3, 4, 0]])) \
                         + self.segmentEmbedding(segmentInfo)

        encoderSelfAttentionMask = getAttentionPadMask(encoderInputs, encoderInputs)

        encoderSelfAttentions = []
        for layer in self.layers:
            encoderOutputs, encoderSelfAttention = layer(encoderOutputs, encoderSelfAttentionMask)
            encoderSelfAttentions.append(encoderSelfAttention)
        return encoderOutputs, encoderSelfAttentions
