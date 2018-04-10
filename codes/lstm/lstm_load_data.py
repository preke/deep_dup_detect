import pandas as pd
import numpy as np
import os
import json
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import pickle

quora_path = '../data/quora_duplicate_questions.tsv'
quora_train_path = '../data/train.tsv'
quora_vali_path = '../data/dev.tsv'
quora_test_path = '../data/test.tsv'

def gen_iter(path, text_field, label_field, pair_id_field, args):
    '''
        Load TabularDataset from path,
        then convert it into a iterator
        return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(
                            path=path,
                            format='tsv',
                            skip_header=True,
                            fields=[
                                    ('label', label_field),
                                    ('question1', text_field),
                                    ('question2', text_field),
                                    ('pair_id', pair_id_field)])

    tmp_iter = data.Iterator(
                    tmp_data,
                    batch_size=args.batch_size,
                    device=0,
                    repeat=False)
    return tmp_data, tmp_iter

def load_quora(args):
    '''
        load as pairs
    '''
    text_field    = data.Field(sequential=True, use_vocab=True, batch_first=True)
    label_field   = data.Field(sequential=False)
    pair_id_field = data.Field(sequential=False)
    
    train_data, train_iter = gen_iter(quora_train_path, text_field, label_field, pair_id_field, args)
    vali_data, vali_iter   = gen_iter(quora_vali_path, text_field, label_field, pair_id_field, args)
    test_data, test_iter   = gen_iter(quora_test_path, text_field, label_field, pair_id_field, args)
    
    text_field.build_vocab(train_data, vali_data)
    label_field.build_vocab(train_data, vali_data)

    return text_field, label_field, \
        train_data, train_iter,\
        vali_data, vali_iter,\
        test_data, test_iter    
