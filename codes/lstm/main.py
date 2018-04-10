# coding = utf-8

import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
import os
import datetime
import traceback
from lstm_load_data import load_quora
from lstm_load_data import load_glove_as_dict
from model import lstm
from model import lstm_similarity

'''
    考虑如何解决两句话很相似，但是仅仅因为一两个字不同导致不是一个label为重复的pair?
    损失函数 or 相似度函数 由两部分组成：
      1. lstm(bilstm) 的hidden layer
      2. bag of words 的 差集
      调权重
'''


parser = argparse.ArgumentParser(description='')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')

# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

args = parser.parse_args()
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))   
args.pretrained_weight = load_glove_as_dict('../../data/wordvec.txt')
args.word_Embedding = True
'''
    begin
'''

text_field, label_field, train_data, train_iter,\
    vali_data, vali_iter, test_data, test_iter = load_quora(args)

text_field.build_vocab(train_data, vali_data, min_freq=5)
label_field.build_vocab(train_data, vali_data)

args.word_embedding_num = len(text_field.vocab)
args.word_embedding_length = 300

lstm_sim = lstm_similarity(args)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
if args.cuda:
    torch.cuda.set_device(args.device)
    lstm_sim = lstm_sim.cuda()

if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    lstm_sim.load_state_dict(torch.load(args.snapshot))
else:
    train(train_iter=train_iter, vali_iter=vali_iter, model=lstm_sim, args=args)





