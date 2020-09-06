# ref https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b

import matplotlib.pyplot as plt
import pandas as pd
import torch

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns

## Preprocessing

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model parameters
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# field
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False,
                   tokenize=tokenizer.encode,
                   lower=False,
                   include_lengths=False,
                   fix_length=MAX_SEQ_LEN,
                   pad_token=PAD_INDEX,
                   unk_token=UNK_INDEX)
fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]

# TabularDataset
source_folder = 'Data'
device = 'cuda'
dest_folder = 'Model'
train, valid, test = TabularDataset.splits(path=source_folder,
                                           train='train.csv',
                                           validation='valid.csv',
                                           test='test.csv',
                                           format='CSV',
                                           fields=fields,
                                           skip_header=True)

# Iterators
train_iter = BucketIterator(train,
                            batch_size=16,
                            sort_key=lambda x: len(x.text),
                            device=device,
                            train=True,
                            sort=True,
                            sort_within_batch=True)

valid_iter = BucketIterator(valid,
                            batch_size=16,
                            sort_key=lambda x: len(x.text),
                            device=device,
                            train=True,
                            sort=True,
                            sort_within_batch=True)

test_iter = BucketIterator(test,
                           batch_size=16,
                           device=device,
                           train=False,
                           shuffle=False,
                           sort=False)

# BERT
## use official classifer
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained(options_name)
        self.linear = nn.Linear(768, 1)
        self.lossFunc = nn.BCELoss()
    def forward(self, text, label):
        label = label.unsqueeze(1).type(torch.float)
        encoding = self.encoder(text)[1] # output at 1 will be pooled output (batchSize, hiddenSize)
        prob = torch.sigmoid(self.linear(encoding))
        loss = self.lossFunc(prob, label)

        #another implementation
        #batchSize = label.size(0)
        #loss = -torch.sum(label * torch.log(prob) + (1 - label) * torch.log(1 - prob)) / batchSize

        text_fea = torch.cat((1 - prob, prob), dim = 1)
        return loss, text_fea


def save_checkpoint(save_path, model, valid_loss):
    state_dict = {'model_state_dict': model.state_dict(),
                'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved in {save_path}')

def load_checkpoint(load_path, model):
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    print(f'Model load in {load_path}')
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')

def load_metrics(load_path):
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']



# define training functions
def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          train_loader=train_iter,
          valid_loader=valid_iter,
          num_epochs=10,
          eval_every=len(train_iter) // 2,
          file_path=dest_folder,
          best_valid_loss=float('inf')):
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, title, text, titletext), _ in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            titletext = torch.transpose(titletext.type(torch.LongTensor), 0, 1)
            titletext = titletext.to(device)
            output = model(titletext, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                y_pred = []
                y_true = []
                with torch.no_grad():

                    # validation loop
                    for (labels, title, text, titletext), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        titletext = torch.transpose(titletext.type(torch.LongTensor), 0, 1)
                        titletext = titletext.to(device)
                        output = model(titletext, labels)
                        loss, outProb = output
                        y_pred.extend(torch.argmax(outProb, 1).tolist())
                        y_true.extend(labels.tolist())
                        valid_running_loss += loss.item()
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))
                print('acurracy score: ', accuracy_score(y_true, y_pred))
                # checkpoint, save better results
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

train(model=model, optimizer=optimizer)


# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, title, text, titletext), _ in test_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            titletext = torch.transpose(titletext.type(torch.LongTensor), 0, 1)
            titletext = titletext.to(device)
            output = model(titletext, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])


best_model = BERT().to(device)

load_checkpoint(dest_folder + '/model.pt', best_model)

evaluate(best_model, test_iter)