import pandas as pd
import numpy as np
import os
import json
import torch
import torchtext.data as data
import torchtext.datasets as datasets
from load_data import gen_iter, load_quora
import argparse


parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()

'''
    Quora:

'''





text_field, label_field, train_data, train_iter,
    vali_data, vali_iter, test_data, test_iter = load_quora()


args.embedding_num = len(TEXT.vocab)
 
