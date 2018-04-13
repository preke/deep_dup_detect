# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt

class CNN_clsm(nn.Module):
    
    def __init__(self, args, wordvec_matrix):
        super(CNN_clsm, self).__init__()
        self.args = args
        
        Ci = 1 # Channel in
        Co = args.kernel_num # 300
        K  = args.kernel_size # 3
        Ss = args.sementic_size
        
        V  = args.embedding_num
        D  = args.embedding_length

        self.embedding = nn.Embedding(V, D)
        self.embedding.weight.data.copy_(wordvec_matrix)
        self.embedding.weight.requires_grad = False
        self.conv    = nn.Conv2d(Ci, Co, (K, D))
        self.dropout = nn.Dropout(args.dropout, self.training)
        self.fc      = nn.Linear(Co, Ss)

    def conv_and_pool(self, sentences_batch):
        '''
            The input is a word_hashed sentences matrix.
            N    : Batch size
            L    : sentences_length
            wh_l : word_hashing length
            K    : kernel size
        '''
        sentences_batch = sentences_batch.unsqueeze(1)
        sentences_batch = F.tanh(self.conv(sentences_batch)).squeeze(3)
        sentences_batch = F.max_pool1d(sentences_batch, sentences_batch.size(2)).squeeze(2)
        # sentences_batch = torch.cat(sentences_batch, 1)
        sentences_batch = self.fc(sentences_batch)
        return sentences_batch        

    def forward(self, query):
        query   = self.embedding(query)
        query   = self.conv_and_pool(query)
        return query

class CNN_Sim(nn.Module):
    def __init__(self, args, wordvec_matrix):
        super(CNN_Sim, self).__init__()
        self.cnn1 = CNN_clsm(args, wordvec_matrix)
        self.cnn2 = CNN_clsm(args, wordvec_matrix)
    def forward(self, q1, q2):
        q1 = self.cnn1.forward(q1)
        q2 = self.cnn2.forward(q2)
        cos_ans = F.cosine_similarity(q1, q2)
        # ignore Gamma
        return cos_ans

