import warnings
warnings.filterwarnings('ignore')
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from nltk.tokenize import word_tokenize ,sent_tokenize
from collections import Counter
from keras.preprocessing.sequence import pad_sequences

from io import open

import torch

from util import config
import numpy as np



"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datasets import load_dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')
train = dataset['train']
val = dataset['validation']
article_train = train['article']
resume_train = train['highlights']
idi_train = train['id']
article_val = val['article']
resume_val = val['highlights']
idi_val = val['id']
article =article_train + article_val
resume = resume_train + resume_val
article = article[:1000]
resume = resume[:1000]
vocab_size = 200000
"""

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
        
START_DECODING = "[SOS]"
STOP_DECODING =  "[EOS]"
PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"

class Vocab:
    def __init__(self, vocab_file, name):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS
        self.oovs = []

        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self.word2index[w] = self.n_words
            self.index2word[self.n_words] = w
            self.n_words += 1

        
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
                if config.vocab_size != 0 and self.n_words >= config.vocab_size:
                     break

    def genrate_input(self , article , resume):
        input=[self.ArticleToindex(art) for art in article]
        resume_out = [self.resumeToindex(res) for res in resume]
        leng_max = max([len(art) for art in input])
        len_max = max([len(res) for res in resume_out])
        input = pad_sequences(input, padding='post' ,maxlen=leng_max)
        resume_out = pad_sequences(resume_out, padding='post' ,maxlen=len_max)
        return input , resume_out


    def word2id(self, word):
        if word not in self.word2index:
            return  self.word2index["UNK"]
        return self.word2index[word]

    def id2word(self, word_id):
        if word_id not in self.index2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.index2word[word_id]


    def outputids2words(self,id_list):
        words = []
        for i in id_list:
            try:
                w = self.id2word(i) # might be [UNK]
            except ValueError as e: # w is OOV
                assert self.oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
                article_oov_idx = i + config.vocab_size
                try:
                    w = self.oovs[article_oov_idx]
                except ValueError as e: # i doesn't correspond to an article oov
                    raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(self.oovs)))
            words.append(w)
        return words


class Article:
    def __init__(self , resume , article , vocab):
        self.resume = resume
        self.article = article
        self.oovs = {}
        self.vocab = vocab
        self.enc_input_length = min(config.max_enc_steps ,len(self.article.split(" ")))
        self.dec_input_length = min(config.max_dec_steps ,len(self.resume.split(" ")))
        self.enc_input = pad_sequences([self.ArticleToindex()], padding='post' , truncating='post',maxlen=config.max_enc_steps,value=self.vocab.word2index['PAD'])[0]
        self.target = pad_sequences([self.resumeToindex()], padding='post', truncating='post' ,maxlen=config.max_dec_steps ,value=self.vocab.word2index['PAD'])[0]
        self.enc_input_resume = [self.vocab.word2id(w) for w in self.resume.split(" ")]

        self.dec_input  = pad_sequences([[self.vocab.word2index['SOS']] + self.resumeToindex() + [self.vocab.word2index['EOS']]], truncating='post',padding='post' ,maxlen=config.max_dec_steps ,value=self.vocab.word2index['PAD'])[0]

    def ArticleToindex(self):
        ids =[]
        for word in word_tokenize(self.article.lower()):
            if(word not in self.vocab.word2index ):
                if word not in self.oovs:
                    self.oovs[word] = self.vocab.n_words + len(self.oovs)
                ids.append(self.oovs[word])
            else:
                ids.append(self.vocab.word2index[word])
        return ids
    
    def resumeToindex(self):
            ids = []
            for word in word_tokenize(self.resume.lower()):
                if(word not in self.vocab.word2index ):
                    if word in self.oovs: # If w is an in-article OOV
                            vocab_idx = self.vocab.n_words + self.oovs[word] # Map to its temporary article OOV number
                            ids.append(vocab_idx)
                    else: # If w is an out-of-article OOV
                        ids.append(3) # Map to the UNK token id
                else:
                     ids.append(self.vocab.word2index[word])
            return ids


class Batch():

    def __init__(self, batch_list_article):

        self.batch_size = len(batch_list_article)
        self.list_article =  batch_list_article
        self.init_enc_batch()
        self.init_decoder_batch()
        self.extra_zeros = None


    def init_enc_batch(self):

            self.enc_batch =np.zeros((self.batch_size, config.max_enc_steps), dtype=np.int32)
            self.enc_batch_lens = np.zeros((self.batch_size), dtype=np.int32)
            self.enc_padding_mask = np.zeros((self.batch_size, config.max_enc_steps), dtype=np.float32)
            for i, article in enumerate(self.list_article):
                self.enc_batch[i, :] = article.enc_input
                self.enc_batch_lens[i] = article.enc_input_length

                for j in range(article.enc_input_length):
                    self.enc_padding_mask[i][j] = 1
            self.enc_batch =  torch.from_numpy(self.enc_batch )
            self.enc_padding_mask =  torch.from_numpy(self.enc_padding_mask)



    def init_decoder_batch(self):

        self.dec_batch =np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_batch_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)


        for i, article in enumerate(self.list_article):
            self.dec_batch[i, :] = article.dec_input
            self.dec_batch_lens[i] = article.dec_input_length
            self.target_batch[i, :] = article.target[:]


        self.dec_batch =  torch.from_numpy(self.dec_batch )



class Batcher(object):

    def __init__(self, vocab, resumes, articles):
        self.batch_size = config.batch_size
        self.count = 0
        self.vocab = vocab
        self.resumes = resumes
        self.articles = articles

    def next_batch(self):
        list_batch = [Article(self.resumes[self.count*self.batch_size + i] , self.articles[self.count*self.batch_size + i] , self.vocab) for i in range(self.batch_size)]
        self.count += 1
        return Batch(list_batch)
