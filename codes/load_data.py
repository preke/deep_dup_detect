import pandas as pd
import numpy as np
import os
import json
import torch
import torchtext.data as data
import torchtext.datasets as datasets

quora_path = '../data/quora_duplicate_questions.tsv'
quora_train_path = '../data/quora_train.csv'
quora_vali_path = '../data/quora_vali.csv'
quora_test_path = '../data/quora_test.csv'


def gen_iter(path, text_field, label_field):
    '''
        Load TabularDataset from path,
        then convert it into a iterator
        return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(path=path
                                    format='CSV',
                                    fields=[
                                    ('ques1', text_field),
                                    ('ques2', text_field),
                                    ('label', label_field)])

    tmp_iter = data.Iterator(
                    tmp_data,
                    batch_size=args.batch_size,
                    device=0,
                    repeat=False)
    return tmp_iter, tmp_data

def load_quora(quora_path=quora_path):
    '''
        load as pairs
    '''
    # Split train-vali-test manually. 
    df_quora = pd.read_csv(quora_path, sep='\t')
    df_quora = df_quora[['question1', 'question2', 'is_duplicate']]
    
    length   = len(df_quora)
    df_train = df_quora.iloc[:length*0.7,:]
    df_vali  = df_quora.iloc[length*0.7:length*0.8,:]
    df_test  = df_quora.iloc[length*0.8:,:]
    
    df_train.to_csv(quora_train_path)
    df_vali.to_csv(quora_vali_path)
    df_test.to_csv(quora_test_path)

    text_field  = data.Field(sequential=True, use_vocab=True, batch_first=True)
    label_field = data.Field(sequential=False)
    
    train_data, train_iter = gen_iter(quora_train_path, text_field, label_field)
    vali_data, vali_iter   = gen_iter(quora_vali_path, text_field, label_field)
    test_data, test_iter   = gen_iter(quora_test_path, text_field, label_field)
    
    text_field.build_vocab(train_data, vali_data)
    return text_field, label_field, \
        train_data, train_iter,\
        vali_data, vali_iter,\
        test_data, test_iter    
    