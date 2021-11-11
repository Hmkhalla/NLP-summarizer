#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib notebook
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize ,sent_tokenize

from collections import Counter

from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### I. Import dataset 

# In[3]:


from datasets import load_dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')


# In[4]:


train = dataset['train']
val = dataset['validation']


# In[5]:


article_train = train['article']
resume_train = train['highlights']
idi_train = train['id']


article_val = val['article']
resume_val = val['highlights']
idi_val = val['id']


# In[6]:


article =article_train + article_val
resume = resume_train + resume_val


# In[7]:


article = article[:1000]
resume = resume[:1000]


# In[8]:


vocab_size = 200000

class Makedata:
        
    def concacate(self , article , resume):
        article = [art.lower() for art in article]
        resume =[art.lower() for art in resume]
        #for i in tqdm(range(len(article))):
            #art = art + article[i] + resume[i]
        a = article + resume
        return ' '.join(a)

    def createVocab(self , article  ,resume, size , filename):
        
        file = open(filename,"a",encoding="utf-8")
        word_freq = Counter(word_tokenize(self.concacate(article , resume)))
        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]
        count = 0
        for word in top_k_words:
            if count < size:
                if not (word_freq[word] == 1):
                    file.write(word+" \n")
                    count+=1
        return 
        


class Vocab:
    def __init__(self, vocab_file, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "PAD", 1:"SOS", 2: "EOS" ,3 : "UNK" }
        self.n_words = 4  # Count SOS and EOS
        self.oovs = []
        
        
        with open(vocab_file, 'r' ,encoding="utf-8") as vocab_f:
              for line in vocab_f:
                pieces = line
                w = pieces.split(" ")[0]
                if w in ["UNK", "SOS" , "EOS"]:
                    raise Exception('[UNK], [SOS] and [EOS] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self.word2index:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self.word2index[w] = self.n_words
                self.index2word[self.n_words] = w
                self.n_words += 1
                if vocab_size != 0 and self.n_words >= vocab_size:
                     break
            
    def ArticleToindex(self , article):
        ids =[]
        for word in word_tokenize(article.lower()):
            if(word not in self.word2index ):
                if word not in self.oovs:
                    self.oovs.append(word)
                oov_num = self.oovs.index(word)
                ids.append(self.n_words + oov_num)
            else:
                ids.append(self.word2index[word])
        return ids
    
    def resumeToindex(self, resume):
            ids = []
            for word in word_tokenize(resume.lower()):
                if(word not in self.word2index ):
                    if word in self.oovs: # If w is an in-article OOV
                            vocab_idx = self.n_words + self.oovs.index(word) # Map to its temporary article OOV number
                            ids.append(vocab_idx)
                    else: # If w is an out-of-article OOV
                        ids.append(2) # Map to the UNK token id
                else:
                     ids.append(self.word2index[word])
            return ids
    
    def genrate_input(self , article , resume):
        input=[self.ArticleToindex(art) for art in article]
        resume_out = [self.resumeToindex(res) for res in resume]
        leng_max = max([len(art) for art in input])
        len_max = max([len(res) for res in resume_out])
        input = pad_sequences(input, padding='post' ,maxlen=leng_max) 
        resume_out = pad_sequences(resume_out, padding='post' ,maxlen=len_max) 
        return input , resume_out

