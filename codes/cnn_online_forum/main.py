# coding = utf-8
import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
from load_data import load_quora, load_glove_as_dict
import mydatasets
import os
import datetime
import traceback
import model
from train import train, test


glove_path = '../../data/wordvec.txt'

# 参数设置：
parser = argparse.ArgumentParser(description='')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=256, help='batch size for training [default: 64]')
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
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=True, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load data
text_field, label_field, train_data, train_iter,\
    vali_data, vali_iter, test_data, test_iter = load_quora(args)

text_field.build_vocab(train_data, vali_data, min_freq=5)
# label_field.build_vocab(train_data, vali_data)

args.embed_num = len(text_field.vocab)
args.embed_dim = 300
args.word_Embedding = True

embedding_dict = load_glove_as_dict(glove_path)
word_vec_list = []
for idx, word in enumerate(text_field.vocab.itos):
    if word in embedding_dict:
        try:
            vector = np.array(embedding_dict[word], dtype=float).reshape(1, args.embed_dim)
        except:
            vector = np.random.rand(1, args.embed_dim)
    else:
        vector = np.random.rand(1, args.embed_dim)
    word_vec_list.append(torch.from_numpy(vector))
wordvec_matrix = torch.cat(word_vec_list)

args.pretrained_weight = wordvec_matrix
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]


cnn = model.CNN_Text(args)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda

if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))
    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()
else:
    try:
        train(train_iter=train_iter, vali_iter=vali_iter, model=cnn, args=args)
    except KeyboardInterrupt:
        print(traceback.print_exc())
        print('\n' + '-' * 89)
        print('Exiting from training early')

test(test_iter=test_iter, model=cnn, args=args)








