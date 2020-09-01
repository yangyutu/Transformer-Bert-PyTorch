import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ffn):
        super(FeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ffn, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ffn, out_channels=d_model, kernel_size=1)
        self.relu = nn.ReLU()
        self.layerNorm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.layerNorm(output + residual)
        return output