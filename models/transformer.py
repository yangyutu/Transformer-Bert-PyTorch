import numpy as np
import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, d_K, d_V, n_heads, n_layers, sourceVocabSize, sourceLength, targetVocabSize, targetLength):
        super(Transformer, self).__init__()
        self.encoder = Encoder(sourceVocabSize, sourceLength, d_model, d_ff, d_K, d_V, n_heads, n_layers)
        self.decoder = Decoder(targetVocabSize, targetLength, d_model, d_ff, d_K, d_V, n_heads, n_layers)
        self.projection = nn.Linear(d_model, targetVocabSize, bias = False)

    def forward(self, encoderInputs, deconderInputs):
        encoderOutputs, encoderSelfAttentions = self.encoder(encoderInputs)
        decoderOutputs, decoderSelfAttentions, deconderEncoderAttentions = self.decoder(deconderInputs,
                                                                                        encoderInputs,
                                                                                        encoderOutputs)
        decoderLogits = self.projection(decoderOutputs)
        return decoderLogits.view(-1, decoderLogits.size(-1)), encoderSelfAttentions, decoderSelfAttentions, deconderEncoderAttentions

