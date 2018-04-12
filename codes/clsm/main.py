import pandas as pd
import numpy as np
import os
import json
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
from clsm_load_data import clsm_gen_question_set
from word_hashing import WordHashing
import pickle
import datetime
import traceback
from train import train, test
from model import CNN_clsm

Train_path = '../../data/clsm_qoura_train_text.tsv'
Vali_path  = '../../data/clsm_qoura_vali_text.tsv'
Test_path  = '../../data/clsm_qoura_test_text.tsv'
embedding_dict_path = 'model/embedding_dict.save'
embedding_length_path = 'model/embedding_length.save'

parser = argparse.ArgumentParser(description='')

# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=100, help='batch size for training [default: 64]')
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
parser.add_argument('-kernel-num', type=int, default=300, help='number of each kind of kernel')
parser.add_argument('-kernel-size', type=str, default=3, help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')


args   = parser.parse_args()

args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.sementic_size = 128 

'''
    Quora & clsm

'''
if not os.path.exists(embedding_dict_path):
    print('Re-create datasets...')
    clsm_gen_question_set()
else:
    print('Load datasets...')
with open(embedding_dict_path, 'rb') as f:
    embedding_dict = pickle.load(f)
with open(embedding_length_path, 'rb') as f:
    embedding_length = pickle.load(f)

TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True)
label_field  = data.Field(sequential=False)
train_data = data.TabularDataset(path=Train_path, 
                                 format='TSV',
                                 skip_header=True,
                                 fields=[('query', TEXT), ('pos_doc', TEXT), ('neg_doc_1', TEXT), 
                                        ('neg_doc_2', TEXT), ('neg_doc_3', TEXT), ('neg_doc_4', TEXT),
                                        ('neg_doc_5', TEXT) ])
vali_data = data.TabularDataset(path=Vali_path, 
                                 format='TSV',
                                 skip_header=True,
                                 fields=[('query', TEXT), ('pos_doc', TEXT), ('neg_doc_1', TEXT), 
                                        ('neg_doc_2', TEXT), ('neg_doc_3', TEXT), ('neg_doc_4', TEXT),
                                        ('neg_doc_5', TEXT) ])
TEXT.build_vocab(train_data, vali_data)

train_iter = data.Iterator(
    train_data,
    batch_size=args.batch_size,
    device=0,
    repeat=False)
vali_iter = data.Iterator(
    vali_data, 
    batch_size=args.batch_size,
    device=0,
    repeat=False)

print('Building vocabulary done. vocabulary length: %s.\n' %str(len(train_data)))
args.embedding_length = embedding_length
args.embedding_num    = len(TEXT.vocab)
print('word vector length: %s.\n' %str(args.embedding_length))

word_vec_list = []
for idx, word in enumerate(TEXT.vocab.itos):
    if word in embedding_dict:
        vector = np.array(embedding_dict[word], dtype=float).reshape(1, embedding_length)
    else:
        vector = np.random.rand(1, args.embedding_length)
    word_vec_list.append(torch.from_numpy(vector))
wordvec_matrix = torch.cat(word_vec_list)

cnn       = CNN_clsm(args, wordvec_matrix)

args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))
else:
    train(train_iter=train_iter, vali_iter=vali_iter, model=cnn, args=args)

'''
    test
'''


test_data = data.TabularDataset( path=Test_path, 
                                 format='TSV',
                                 skip_header=True,
                                 fields=[('query', TEXT), ('doc', TEXT), ('label', label_field)])
label_field.build_vocab(test_data)

for idx, word in enumerate(TEXT.vocab.itos):
    print('%s: %s' %(idx, word))



test_iter = data.Iterator(
    test_data,
    batch_size=args.batch_size,
    device=0,
    repeat=False)

test(test_iter=test_iter, model=cnn, args=args)





