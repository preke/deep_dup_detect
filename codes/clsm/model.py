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

    def forward(self, query, doc):
        '''
            Input a query and doc,
            return the similarity in sementic layer
        '''
        query   = self.embedding(query)
        # query   = Variable(query)
        doc     = self.embedding(doc)
        # doc     = Variable(doc)
        query   = self.conv_and_pool(query)
        doc     = self.conv_and_pool(doc)
        gamma   = 0.1 # Variable(torch.FloatTensor()) # smoothing factor
        # gamma   = gamma.cuda() if self.args.cuda else gamma
        cos_sim = gamma * F.cosine_similarity(query, doc)
        return cos_sim