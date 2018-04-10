import pandas as pd
import numpy as np
import os
import json
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import pickle
# from word_hashing import WordHashing

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
                                    ('num', pair_id_field)])

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
    # Split train-vali-test manually. 
    # df_quora = pd.read_csv(quora_path, sep='\t')
    # df_quora = df_quora[['question1', 'question2', 'is_duplicate']]
    
    # length   = len(df_quora)
    # df_train = df_quora.iloc[:int(length*0.7),:]
    # df_vali  = df_quora.iloc[int(length*0.7):int(length*0.8),:]
    # df_test  = df_quora.iloc[int(length*0.8):,:]
    
    
    # df_train.to_csv(quora_train_path, index=False, sep='\t')
    # df_vali.to_csv(quora_vali_path, index=False, sep='\t')
    # df_test.to_csv(quora_test_path, index=False, sep='\t')
    
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


def clsm_gen_question_set():
    df = pd.read_csv(quora_path, sep='\t')
    df_pairs = df[['qid1', 'qid2', 'is_duplicate']]
    df_pos = df_pairs[df_pairs['is_duplicate'] == 1]
    
    df_dup = []
    for i, r in df_pos.iterrows():
        if r['qid1'] < r['qid2']:
            df_dup.append([r['qid1'], r['qid2'], 1])
    df_dup = pd.DataFrame(df_dup, columns=['qid1', 'qid2', 'is_duplicate'])
    print 'df_dup shape: ', df_dup.shape
    df_dup.to_csv('../data/clsm_quora_dup.tsv', sep='\t', index=False)

    length = len(df_dup)
    df_train = df_dup.iloc[:int(length*0.95), :]
    df_test = df_dup.iloc[int(length*0.95):, :]
    
    df_train_new = []
    for i, r in df_train.iterrows():
        temp_list = [r['qid1'], r['qid2']]
        j = 0
        while(True):
            tmp = np.random.choice(df_train['qid2'].values)
            if tmp != r['qid1'] and tmp != r['qid2']:
                j += 1
                temp_list.append(tmp)
            if j >= 5:
                break
        df_train_new.append(temp_list)
    df_train_new = pd.DataFrame(df_train_new, columns=['query', 'pos_doc', 'neg_doc_1', 'neg_doc_2', 'neg_doc_3', 'neg_doc_4', 'neg_doc_5'])
    df_vali = df_train_new.iloc[int(len(df_train_new)*0.95):, :]
    df_train = df_train_new.iloc[:int(len(df_train_new)*0.95), :]

    df_train.to_csv('../data/clsm_qoura_train.tsv', sep='\t', index=False)
    df_vali.to_csv('../data/clsm_qoura_vali.tsv', sep='\t', index=False)
    df_test.to_csv('../data/clsm_qoura_test.tsv', sep='\t', index=False)

    ques_dict = {}
    for i, r in df.iterrows():
        ques_dict[r['qid1']] = r['question1']
        ques_dict[r['qid2']] = r['question2']

    corpus = []
    for i, r in df_train_new.iterrows():
        corpus += ques_dict[r['query']].split(' ')
        corpus += ques_dict[r['pos_doc']].split(' ')
        corpus += ques_dict[r['neg_doc_1']].split(' ')
        corpus += ques_dict[r['neg_doc_2']].split(' ')
        corpus += ques_dict[r['neg_doc_3']].split(' ')
        corpus += ques_dict[r['neg_doc_4']].split(' ')
        corpus += ques_dict[r['neg_doc_5']].split(' ')

    wh_instance      = WordHashing(corpus)
    embedding_length = 0
    embedding_dict   = {}
    for word in corpus:
        embedding_dict[word] = wh_instance.hashing(word)
        embedding_length     = len(wh_instance.hashing(word))

    
    return embedding_dict, embedding_length
    







