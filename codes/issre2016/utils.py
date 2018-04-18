# coding = utf-8
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from math import sqrt
from math import log
import traceback
import datetime

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import pickle

model_path = 'model/'

class Preprocess(object):
    def __init__(self):
        self.english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'']  
        self.stop = set(stopwords.words('english'))

    def punctuate(self, text):
        ans = ""
        for letter in text:
            if letter in self.english_punctuations:
                ans += ' '
            else:
                ans += letter
        return ans

    def stem_and_stop_removal(self, text):
        text = self.punctuate(text)
        word_list = word_tokenize(text)
        lancaster_stemmer = LancasterStemmer()
        word_list = [lancaster_stemmer.stem(i) for i in word_list]
        word_list = [i for i in word_list if i not in self.stop]
        return word_list

def train_word2vec_model(df):
    corpus = []
    for i, r in df.iterrows():
        corpus.append(r['Title'])
    word2vec_model = Word2Vec(corpus, size=100, window=3, min_count=1, sg=1)
    return word2vec_model

def get_tfidf_dict(df):
    corpus = [' '.join(i) for i in df['Title']]
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    words=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    tf_idf_dict = {}
    for i in range(len(weight)):
        tf_idf_dict[df['Issue_id'][i]] = weight[i].reshape(1,-1)
    
    return tf_idf_dict



def load_data(data_path):
    preprocess = Preprocess()
    df = pd.read_csv(open(data_path, 'rU'), encoding='utf-8')
    df = df[['Issue_id', 'Component', 'Title', 'Duplicated_issue']]
    df['Title'] = df['Title'].apply(preprocess.stem_and_stop_removal)
    tf_idf_dict = {}# get_tfidf_dict(df)
    df['is_duplicate'] = df['Duplicated_issue'].apply(lambda x: not pd.isnull(x))
    word2vec_model = train_word2vec_model(df)
    return df, word2vec_model, tf_idf_dict





















