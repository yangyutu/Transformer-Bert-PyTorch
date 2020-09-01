import numpy as np
import torch

def calAngle(position, hidIdx, d_model):
    return position / np.power(10000.0, 2 * (hidIdx // 2) / d_model)

def getPositionAngleVector(position, d_model):
    return [calAngle(position, j, d_model) for j in range(d_model)]

def getSinCosEncoding(n_position, d_model):
    sinCosTable = np.array([getPositionAngleVector(i, d_model) for i in range(n_position)])
    sinCosTable[:, 0::2] = np.sin(sinCosTable[:,0::2])
    sinCosTable[:, 1::2] = np.sin(sinCosTable[:, 1::2])
    return torch.FloatTensor(sinCosTable)

