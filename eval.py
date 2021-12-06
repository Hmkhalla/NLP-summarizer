import os

from tqdm import tqdm

from torch.utils.data import DataLoader
from model import Model

from util import config
from util.data import collate_batch, article_pipeline, resume_pipeline
from datasets import load_dataset
import util.vocab as vocab
from util.vocab import Vocabulary
from util.train_util import *
from numpy import random
import argparse
import wandb
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


class Evale():
    def __init__(self, id):
        dataset = load_dataset('gigaword')
        vocab = Vocabulary(dataset['train'].select(range(config.datasize)), vocab_size=config.vocab_size)
        self.model = Model(0, 3, 2)
        self.model = self.model.to(device)
        load_model_path = os.path.join(config.save_model_path, '0' * (7 - len(str(id))) + str(id) + '.tar')
        checkpoint = torch.load(load_model_path)
        start_iter = checkpoint["iter"]
        self.model.load_state_dict(checkpoint["model_dict"])
        process_batch = lambda batche: collate_batch(batche, vocab)
        val_dataset = dataset['test']
        val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=process_batch)
        self.batch = next(iter(val_data_loader))
        self.true_article, self.true_resume, self.oovs = self.batch
        self.inv_oovs = {self.oovs[key]: key for key in self.oovs.keys()}

    def get_values(self):
        return self.model(self.batch)

    def get_resumes(self):
        resumes = []
        id_resume, dist = self.model(self.batch)
        for resume in id_resume:
            resumes.append(" ".join([vocab.vocab.lookup_token(word_ix) if (word_ix < config.vocab_size) else
                                     self.inv_oovs[word_ix - config.vocab_size] for word_ix in resume]))
        articles = []
        for article in self.true_article[0]:
            articles.append(" ".join([vocab.vocab.lookup_token(word_ix) if (word_ix < config.vocab_size) else
                                      self.inv_oovs[word_ix - config.vocab_size] for word_ix in article]))

        Trueresumes = []
        for resume in self.true_resume[0]:
            Trueresumes.append(" ".join([vocab.vocab.lookup_token(word_ix) if (word_ix < config.vocab_size) else
                                      self.inv_oovs[word_ix - config.vocab_size] for word_ix in resume]))
        return articles, Trueresumes, resumes