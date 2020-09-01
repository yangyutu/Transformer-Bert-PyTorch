import numpy as np
import torch

def getAttentionPadMask(seq_q, seq_k):
    batchSize, len_q = seq_q.size()
    batchSize, len_k = seq_k.size()
    # 0 is special symbol P in key sequence
    padAttentionMask = seq_k.data.eq(0).unsqueeze(1)
    return padAttentionMask.expand(batchSize, len_q, len_k)

def getAttentionSubsequentMask(seq):
    attentionShape = [seq.size(0), seq.size(1), seq.size(1)]
    # get the lower triangle matrix as the matrix
    subsequentMask = np.triu(np.ones(attentionShape), k=1)
    subsequentMask = torch.from_numpy(subsequentMask).bool()
    return subsequentMask