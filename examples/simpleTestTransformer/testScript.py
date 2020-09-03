import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from models.transformer import Transformer

# special symbols are P, S, E, (0, 5, 6)
# here are three sentences. P is padding code, S is starting code, E is ending code
# first sentence is input sequence, second sequence is output sequence (used as the input in decoder side)
# third sentence is target sequence (which is one left shift of output sequence)
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
sourceVocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4}
sourceVocabSize = len(sourceVocab)

targetVocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'S' : 5, 'E' : 6}
number_dict = {i: w for i, w in enumerate(targetVocab)}
targetVocabSize = len(targetVocab)

sourceLength = 5
targetLength = 5

d_model = 512
d_ffn = 2048
d_K = d_V = 64
n_layers = 6
n_heads = 8

def make_batch(sentences):
    input_batch = [[sourceVocab[w] for w in sentences[0].split()]]
    output_batch = [[targetVocab[w] for w in sentences[1].split()]]
    target_batch = [[targetVocab[w] for w in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

print(make_batch(sentences))


model = Transformer(d_model, d_ffn, d_K, d_V, n_heads, n_layers, sourceVocabSize, sourceLength, targetVocabSize, targetLength)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
n_epoch = 20
for epoch in range(n_epoch):
    optimizer.zero_grad()
    encoderInputs, decoderInputs, targetBatch = make_batch(sentences)
    outputs, _, _, _ = model(encoderInputs, decoderInputs)
    loss = criterion(outputs, targetBatch.contiguous().view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

