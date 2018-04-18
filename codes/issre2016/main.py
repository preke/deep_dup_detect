'''
    Use pretained word-embedding vectors
    Pairwise method.
    Just use title.
    ---
    Set the pos-neg ratio to 1:1
    calculate recall@1, recall@5, recall@10
'''

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_data


DATA_PATH = '../../data/Spark.csv'

def get_sim_score1(Issue1, Issue2, tf_idf_dict):
    '''
        tf-idf vectors similarity score.
    '''
    return cosine_similarity(tf_idf_dict[Issue1], tf_idf_dict[Issue2])

def get_sim_score2(title1, title2, w2v_model):
    '''
        word embedding vectors similarity score.
        use pretained vectors.
    '''
    vec1 = np.array([0.0]*100)
    vec2 = np.array([0.0]*100)
    for word in title1:
        vec1 += np.array(w2v_model[word])
    vec1 = vec1 / len(title1)
    for word in title2:
        vec2 += np.array(w2v_model[word])
    vec2 = vec2 / len(title2)
    return cosine_similarity(vec1.reshape(1,-1), vec2.reshape(1,-1))


def get_sim_score3(row1, row2):
    '''
        Product & Component(Here only component)
    '''
    score3 = 0.5 # Already in the same product
    if row1['Component'] == row2['Component'] and row1['Component'] != '':
        score3 += 0.5
    else:
        pass
    return score3


def combine_score(row1, row2, w2v_model, tf_idf_dict):
    score1 = 1# get_sim_score1(row1['Issue_id'], row2['Issue_id'], tf_idf_dict)
    score2 = get_sim_score2(row1['Title'], row2['Title'], w2v_model)
    score3 = get_sim_score3(row1, row2)
    score  = (score1 + score2) * score3
    return score


def evaluation(df_querys, total_scores):
    '''
        calculate recall@1, recall@5, recall@10, mrr, map
    '''
    total_recall_1 = 0
    total_recall_5 = 0
    total_recall_10 = 0
    
    index = 0
    for i, r in df_querys[:1].iterrows():
        dup_list = r['Duplicated_issue'].split(';')
        for issue in total_scores[index][:1]:
            if issue[0] in dup_list:
                total_recall_1 += 1
                break
        for issue in total_scores[index][:5]:
            if issue[0] in dup_list:
                total_recall_5 += 1
                break
        for issue in total_scores[index][:10]:
            if issue[0] in dup_list:
                total_recall_10 += 1
                break
    total_recall_1 /= len(df_querys)
    total_recall_5 /= len(df_querys)
    total_recall_10 /= len(df_querys)
    print('Recall at 1 is %f.' %total_recall_1)
    print('Recall at 5 is %f.' %total_recall_5)
    print('Recall at 10 is %f.' %total_recall_10)

if __name__ == '__main__':
    df_data, word2vec_model, tf_idf_dict = load_data(DATA_PATH)
    df_querys = df_data[df_data['is_duplicate']==True]
    
    total_scores = []
    for i, r1 in df_querys[:1].iterrows():
        scores = []
        for j, r2 in df_data.iterrows():
            if r1['Issue_id'] != r2['Issue_id']:
                score = combine_score(r1, r2, word2vec_model, tf_idf_dict)
                scores.append((r2['Issue_id'], score)) 
        scores = sorted(scores, key=lambda x:x[1], reverse=True)
        total_scores.append(scores)

    evaluation(df_querys, total_scores)




