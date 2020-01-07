#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:20:40 2019

@author: dhanachandra
"""
import torch
import random
import numpy as np
import os
#make to initialize the newtork with same random number
def seed_torch(seed=12345):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

seed_torch()
class Example(object):
    def __init__(self, id, sent1, sent2, sim_score=0, normalized_score=0):
        self.id = id
        self.sent1 = sent1
        self.sent2 = sent2
        self.sim_score = sim_score
        self.normalized_score = normalized_score
    
    def get_id(self):
        return self.id
    
    def get_sent1(self):
        return self.sent1
    
    def get_sent2(self):
        return self.sent2
    
    def get_sim_score(self):
        return self.sim_score
    
    def get_normalized_score(self):
        return self.normalized_score
    
    def to_string(self):
        return {'ID: ': self.id, 'Sent1: ' : self.sent1, 'Sent2: ' : self.sent2, 'Sim score: ': self.sim_score}

def read_example(file):
    examples = []
    id = 1
    with open(file) as reader:
        for line in reader:
            cols = line.split('\t')
            sim_score = float(cols[2])
            example = Example(id, cols[0], cols[1], float(cols[2]))
            examples.append(example)
            id +=1
    return examples

tranining_examples = read_example('clinicalSTS2019.train.txt')
val_examples = read_example('final_validation.txt')

import numpy as np 
from torch.utils import data 
import torch 
from pytorch_pretrained_bert import BertTokenizer
import string

class Param:
    batch_size = 32 
    lr = 1e-4
    n_epochs = 64
    p = 0.3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pad_token = 0
    cls_token = '[CLS]'
    sep_token= '[SEP]'
    vocab_path = '../biobert_v1.0_pubmed_pmc/vocab.txt'
    bert_config = '../biobert_v1.0_pubmed_pmc/bert_config.json'
    bert_weight = '../biobert_v1.0_pubmed_pmc/weight/pytorch_weight'
    sep_sent = True
    num_fold = 5
    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
    # to remove puntuations
class STSDataset(data.Dataset):
    def __init__(self, examples, param):
       self.param = param
       self.examples = examples
       self.replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
       self.tokenizer = param.tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        sent1 = example.get_sent1().translate(self.replace_punctuation)
        sent2 = example.get_sent2().translate(self.replace_punctuation)
        sim_score = example.get_sim_score()
        sent1_tok = self.tokenizer.tokenize(sent1)
        sent2_tok = self.tokenizer.tokenize(sent2)
        #make two inputs, Sent1 Sent2 and Sent2 Sent1
        sent_tokens_f = [self.param.cls_token] + sent1_tok + [self.param.sep_token] + sent2_tok + [self.param.sep_token]
        sent_tokens_r = [self.param.cls_token] + sent2_tok + [self.param.sep_token] + sent1_tok + [self.param.sep_token]
        sent_ids_f = self.tokenizer.convert_tokens_to_ids(sent_tokens_f)
        sent_ids_r = self.tokenizer.convert_tokens_to_ids(sent_tokens_r)
        return sent_ids_f, sent_ids_r, len(sent_ids_f), len(sent_ids_r), sim_score, example.get_id()

def pad(batch, sep_sent = Param.sep_sent):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    sent_id_f_seqlens = f(2)
    sim_scores = f(4)
    exm_ids = f(5)
    maxlen = np.array(sent_id_f_seqlens).max()
    #sent2_maxlen = np.array(sent2_seqlens).max()
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x_f = f(0, maxlen)
    x_r = f(1, maxlen)
    f = torch.LongTensor
    return f(x_f), f(x_r), torch.FloatTensor(sim_scores), exm_ids

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
#from Data_Loader import Param
import torch.nn.functional as F


class STS_NET(nn.Module):
    def __init__(self, config, bert_state_dict, device = Param.device):
        super().__init__()
        self.bert = BertModel(config)
        #print('bert initialized from config')
        if bert_state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)
        self.bert.eval()
        self.dropout = nn.Dropout(p=Param.p)
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768, hidden_size=768//2)
        self.f1 = nn.Linear(768//2, 128)
        self.f2 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)
        self.device = device
        
    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, 768//2).to(self.device), torch.zeros(2, batch_size, 768//2).to(self.device)

    def forward(self, x_f, x_r):
        batch_size = x_f.size()[0]
        x_f = x_f.to(self.device)
        x_r = x_r.to(self.device)
        xf_encoded_layers, _ = self.bert(x_f)
        enc_f = xf_encoded_layers[-1]
        enc = enc_f.permute(1, 0, 2)
        enc = self.dropout(enc)
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.rnn(enc, self.hidden)
        last_hidden_state, last_cell_state = self.hidden
        rnn_out = self.dropout(last_hidden_state)
        f1_out = F.relu(self.f1(last_hidden_state[-1]))
        f2_out = F.relu(self.f2(f1_out))
        out = self.out(f2_out)
        return out

from pytorch_pretrained_bert.modeling import BertConfig
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import os
import numpy as np
from collections import OrderedDict
import string
import subprocess
from tqdm import tqdm
from matplotlib import pyplot as plt 

replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
# prepare biobert dict 

from collections import OrderedDict
tmp_d = torch.load(Param.bert_weight, map_location=Param.device)
state_dict = OrderedDict()
for i in list(tmp_d.keys())[:199]:
    x = i
    if i.find('bert') > -1:
        x = '.'.join(i.split('.')[1:])
    state_dict[x] = tmp_d[i]


def train(model, train_iter, path, epoch, optimizer=None, criterion=None):
    model.train()
    #criterion = F.mse_loss 
    criterion = nn.MSELoss(reduction='sum')

    #optimizer = optim.SGD(model.parameters(), lr = 0.01)
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    optimizer = optim.Adam(model.parameters(), lr=Param.lr)
    t_loss = 0.0
    
    for i, batch in enumerate(train_iter):
        sentf_ids, sentr_ids, sim_score, _= batch
        y_true = sim_score.to(model.device)
        optimizer.zero_grad()
        y_pred = model(sentf_ids, sentr_ids)
        loss = criterion(y_pred, y_true.view(-1, 1))
        t_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i== -1 :
            print("=====sanity check======")
            print("Sent1: ", sent1_ids.cpu().numpy()[0])
            print("Sent2: ", sent2_ids.cpu().numpy()[0])
            print("Sent1 tokens:", Param.tokenizer.convert_ids_to_tokens(sent1_ids.cpu().numpy()[0]))
            print("Sent2 tokens:", Param.tokenizer.convert_ids_to_tokens(sent2_ids.cpu().numpy()[0]))
            print("y:", y_true.cpu().numpy()[0])
            #import sys
            #sys.exit(1)

        if i%10 ==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
    print('Total loss: ', t_loss/len(train_iter))
    save_model(epoch, model, optimizer, t_loss/len(train_iter), path)

    return model, optimizer, t_loss/len(train_iter)

def evaluate(model, test_iter, criterion):
    model.eval()
    Y_predicted = []
    Y_true = []
    exam_ids = []
    t_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter, 1):
            sentf_ids, sentr_ids, sim_score, b_exam_ids = batch
            sim_score = sim_score.to(model.device)
            y_pred = model(sentf_ids, sentr_ids)
            loss = criterion(y_pred, sim_score.view(-1, 1))
            t_loss += loss.item()
            #print(labels.numpy())
            if torch.cuda.is_available():
                Y_true = np.concatenate([Y_true, sim_score.cpu().numpy()])
            else:
                Y_true = np.concatenate([Y_true, sim_score.numpy()])
            for score in y_pred:
                Y_predicted.append(round(score.item(), 4))
            exam_ids = exam_ids + b_exam_ids
    print('Validation loss: ', t_loss/len(test_iter))
    return Y_true, Y_predicted, t_loss/len(test_iter), exam_ids

def predict(tokenizer, model, example):        
    sent1 = example.get_sent1().translate(replace_punctuation)
    sent2 = example.get_sent2().translate(replace_punctuation)
    sim_score = example.get_sim_score()
    sent1_tok = tokenizer.tokenize(sent1)
    sent1_tokens = [Param.cls_token] + sent1_tok + [Param.sep_token]
    sent2_tok = tokenizer.tokenize(sent2)
    sent2_tokens = [Param.cls_token] + sent2_tok + [Param.sep_token]
    sent1_ids = tokenizer.convert_tokens_to_ids(sent1_tokens)
    sent2_ids = tokenizer.convert_tokens_to_ids(sent2_tokens)
        
    if Param.sep_sent:
        x1, x2 = sent1_ids, sent2_ids
    else:
        x1, x2 = sent1_ids + sent2_ids[1:], []
    pred = model(x1, x2)
    y_pred = pred.item()
    return y_pred, sim_score

def save_model(epoch, model, optimizer, loss, path):
    print('Saving the model for %d in the path: ' %(epoch), end=" ")
    print(path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, path+'/model_'+str(epoch))

def load_from_checkpoint(path):
    config = BertConfig(vocab_size_or_config_json_file=Param.bert_config)
    model = STS_NET(config = config, bert_state_dict = state_dict, device=Param.device)
    optimizer = optim.Adam(model.parameters(), lr=Param.lr)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

train_dataset = STSDataset(tranining_examples, Param)
eval_dataset = STSDataset(val_examples, Param)
        # Define model 
path = '../models/lstm_forwback_fulldataset'
if not os.path.isdir(path):
    os.mkdir(path)
        #print('Loading bert model')
config = BertConfig(vocab_size_or_config_json_file=Param.bert_config)
model = STS_NET(config=config, bert_state_dict = state_dict) 
        
    
if torch.cuda.is_available():
    model.cuda()
        #model.train()
        # update with already pretrained weight
train_iter = data.DataLoader(dataset=train_dataset,
                            batch_size=Param.batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=pad)
eval_iter = data.DataLoader(dataset=eval_dataset,
                            batch_size=Param.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=pad)
x_axis = np.arange(1, Param.n_epochs+1)
y_axis_train = []
y_axis_test = []
pearson_score = []
MAX_SCORE = 0.0
MAX_EPOCH = 0
for epoch in tqdm(range(1, Param.n_epochs+1)):
    model, optimizer, loss = train(model, train_iter, path, epoch)
    y_axis_train.append(loss)
    criterion = nn.MSELoss(reduction='sum')
    Y_true, Y_pred, t_loss, examp_ids = evaluate(model, eval_iter, criterion)
    y_axis_test.append(t_loss)
    writer1 = open('true_score', 'w')
    writer2 = open('pred_score', 'w')
    writer3 = open('fold_'+str(fd)+'/prediction'+str(epoch), 'w')
    print("=========eval at epoch={%d}=========" %(epoch))
    for i, _ in enumerate(Y_true):
        #exm = id_2_example[examp_ids[i]]
        writer1.write(str(Y_true[i]) +'\n')
        writer2.write(str(Y_pred[i]) +'\n')
        writer3.write(val_examples[i].get_sent1() + '\t' + val_examples[i].get_sent2() 
                    + '\t' + str(Y_true[i]) + '\t' + str(Y_pred[i]) +'\n')
    test = subprocess.Popen(["./correlation-noconfidence.pl", "true_score", "pred_score"], stdout=subprocess.PIPE)
    output = test.communicate()[0].strip()[:-1]
    print('Result: ', str(output))
    if str(output.decode("utf-8")) == 'Na':
        pearson_score.append(0.0)
        continue
    if len(str(output.decode("utf-8")).strip()) > 1:
        x = float(output)
        print('Result: ', x * 100.0, type(x))
        pearson_score.append(x * 100.0)
        if MAX_SCORE < x * 100.0:
            MAX_SCORE = x * 100.0
            MAX_EPOCH = epoch
                
    else:
        pearson_score.append(0.0)
        
plt.plot(x_axis, y_axis_train, label='Training Loss')
plt.plot(x_axis, y_axis_test, label='Testing Loss')
plt.plot(x_axis, pearson_score, label='Pearson Score')
    
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title("Training Loss Vs Testing Loss and Val_score")
plt.legend()
plt.show()
plt.savefig(path+'Loss_vs_score'+'.png')

print("Highest score is %f at epoch %d" %(MAX_SCORE, MAX_EPOCH))




