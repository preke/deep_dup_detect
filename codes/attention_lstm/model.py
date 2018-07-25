# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np



class DA_lstm(nn.Module):
    def __init__(self, vocab_size, device, word_matrix=None, embed_dim=300, lstm_hidden_dim=200):
        super(DA_lstm, self).__init__()
        self.embed_num      = vocab_size
        self.embed_dim      = embed_dim
        self.word_embedding = nn.Embedding(self.embed_num, self.embed_dim)
        self.device         = device
        
        if word_matrix is not None:
            word_matrix = torch.tensor(word_matrix).to(self.device)
            self.word_embedding.weight.data.copy_(word_matrix)
            self.word_embedding.weight.requires_grad = False

        self.mlp_f = self.mlp_layers(self.embed_dim, self.embed_dim)
        self.mlp_g = self.mlp_layers(2 * self.embed_dim, self.embed_dim)
        self.mlp_h = self.mlp_layers(2 * self.embed_dim, self.embed_dim)
        self.final_linear_1 = self.mlp_layers(self.embed_dim, self.embed_dim)
        self.final_linear_2 = self.mlp_layers(self.embed_dim, self.embed_dim)
        self.final_linear_3 = self.mlp_layers(self.embed_dim, 1)
        
        self.hidden_dim = lstm_hidden_dim
        self.lstm       = nn.LSTM(input_size=self.embed_dim, 
                            hidden_size=self.hidden_dim//2,
                            batch_first=True,
                            bidirectional=True)

        self.linear1 = nn.Linear(3, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.dist    = nn.PairwiseDistance(2)

    
    def lstm_embedding(self, lstm, word_embedding):
        lstm_out,(lstm_h, lstm_c) = lstm(word_embedding)
        seq_embedding = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        return seq_embedding

    def mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())   
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):
        sent1_linear_embedding = self.word_embedding(sent1_linear)
        sent2_linear_embedding = self.word_embedding(sent2_linear)
        len1 = sent1_linear_embedding.size(1)
        len2 = sent2_linear_embedding.size(1)
        
        
        '''Attend'''
        f1 = self.mlp_f(sent1_linear_embedding.view(-1, self.embed_dim))
        f2 = self.mlp_f(sent2_linear_embedding.view(-1, self.embed_dim))
        f1 = f1.view(-1, len1, self.embed_dim)
        f2 = f2.view(-1, len2, self.embed_dim)
        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2)) 
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2
        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        '''Compare''' 
        sent1_combine = torch.cat(
            (sent1_linear_embedding, torch.bmm(prob1, sent2_linear_embedding)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear_embedding, torch.bmm(prob2, sent1_linear_embedding)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''Aggregate'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.embed_dim))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.embed_dim))
        g1 = g1.view(-1, len1, self.embed_dim)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.embed_dim)
        # batch_size x len2 x hidden_size
        
        
        '''lstm'''
        g1 = self.lstm_embedding(self.lstm, g1)
        g2 = self.lstm_embedding(self.lstm, g2)
        
        '''interaction'''
        cosine_sim = F.cosine_similarity(g1, g2).view(-1, 1)
        dot_value  = torch.bmm(
                            g1.view(g1.size()[0], 1, g1.size()[1]), 
                            g2.view(g1.size()[0], g1.size()[1], 1)
                            ).view(g1.size()[0], 1)
        dist_value = self.dist(g1, g2).view(g1.size()[0], 1)

        '''dense layers'''
        result = torch.cat((cosine_sim, dot_value, dist_value), dim=1)

        result = self.linear1(result)
        result = self.dropout(result)
        result = F.relu(result)

        result = self.linear2(result)
        result = self.dropout(result)
        result = F.relu(result)

        result = self.linear3(result)
        # print(result.shape)
        return result.squeeze()

