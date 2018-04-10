import pandas as pd
import numpy as np
from word_hashing import WordHashing
import pickle

def clsm_gen_question_set():
    quora_path = '../../data/quora_duplicate_questions.tsv'
    df = pd.read_csv(quora_path, sep='\t')
    df_pairs = df[['qid1', 'qid2', 'is_duplicate']]
    df_pos = df_pairs[df_pairs['is_duplicate'] == 1]
    
    df_dup = []
    for i, r in df_pos.iterrows():
        if r['qid1'] < r['qid2']:
            df_dup.append([r['qid1'], r['qid2'], 1])
    df_dup = pd.DataFrame(df_dup, columns=['qid1', 'qid2', 'is_duplicate'])
    # print('df_dup shape: ', df_dup.shape)
    df_dup.to_csv('../../data/clsm_quora_dup.tsv', sep='\t', index=False)

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

    df_train.to_csv('../../data/clsm_qoura_train.tsv', sep='\t', index=False)
    df_vali.to_csv('../../data/clsm_qoura_vali.tsv', sep='\t', index=False)
    df_test.to_csv('../../data/clsm_qoura_test.tsv', sep='\t', index=False)

    ques_dict = {}
    for i, r in df.iterrows():
        ques_dict[r['qid1']] = str(r['question1']).lower()
        ques_dict[r['qid2']] = str(r['question2']).lower()
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
    print('embedding_length: %s' %embedding_length)
    # df_train['query_text']     = df_train['query'].apply(lambda x: ques_dict[x])
    # df_train['pos_doc_text']   = df_train['pos_doc'].apply(lambda x: ques_dict[x])
    # df_train['neg_doc_1_text'] = df_train['neg_doc_1'].apply(lambda x: ques_dict[x])
    # df_train['neg_doc_2_text'] = df_train['neg_doc_2'].apply(lambda x: ques_dict[x])
    # df_train['neg_doc_3_text'] = df_train['neg_doc_3'].apply(lambda x: ques_dict[x])
    # df_train['neg_doc_4_text'] = df_train['neg_doc_4'].apply(lambda x: ques_dict[x])
    # df_train['neg_doc_5_text'] = df_train['neg_doc_5'].apply(lambda x: ques_dict[x])
    
    # df_vali['query_text']     = df_vali['query'].apply(lambda x: ques_dict[x])
    # df_vali['pos_doc_text']   = df_vali['pos_doc'].apply(lambda x: ques_dict[x])
    # df_vali['neg_doc_1_text'] = df_vali['neg_doc_1'].apply(lambda x: ques_dict[x])
    # df_vali['neg_doc_2_text'] = df_vali['neg_doc_2'].apply(lambda x: ques_dict[x])
    # df_vali['neg_doc_3_text'] = df_vali['neg_doc_3'].apply(lambda x: ques_dict[x])
    # df_vali['neg_doc_4_text'] = df_vali['neg_doc_4'].apply(lambda x: ques_dict[x])
    # df_vali['neg_doc_5_text'] = df_vali['neg_doc_5'].apply(lambda x: ques_dict[x])
    
    # df_test['ques1_text'] = df_test['qid1'].apply(lambda x: ques_dict[x])
    # df_test['ques2_text'] = df_test['qid2'].apply(lambda x: ques_dict[x])

    # df_train_text = df_train[['query_text', 'pos_doc_text', 'neg_doc_1_text',\
    #                     'neg_doc_2_text', 'neg_doc_3_text', 'neg_doc_4_text', 'neg_doc_5_text']]
    # df_train_text.to_csv('../../data/clsm_qoura_train_text.tsv', sep='\t', index=False)
    
    # df_vali_text = df_vali[['query_text', 'pos_doc_text', 'neg_doc_1_text',\
    #                     'neg_doc_2_text', 'neg_doc_3_text', 'neg_doc_4_text', 'neg_doc_5_text']]
    # df_vali_text.to_csv('../../data/clsm_qoura_vali_text.tsv', sep='\t', index=False)
    
    # df_test_text = df_test[['ques1_text', 'ques2_text', 'is_duplicate']]
    # df_test_text.to_csv('../../data/clsm_qoura_test_text.tsv', sep='\t', index=False)    
    
    # with open('model/embedding_dict.save', 'wb') as f:
    #     pickle.dump(embedding_dict, f)
    # with open('model/embedding_length.save', 'wb') as f:
    #     pickle.dump(embedding_length, f)






