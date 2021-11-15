import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"    #Set cuda device

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model

from util import config, data
#from util.batcher import Batcher
from util.data import Vocab
from util.train_util import *
from torch.distributions import Categorical
from rouge import Rouge
from numpy import random
import argparse

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class Train(object):
    def __init__(self, opt):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.opt = opt
        self.start_id = self.vocab.word2id(data.START_DECODING)
        self.end_id = self.vocab.word2id(data.STOP_DECODING)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        time.sleep(5)

    def save_model(self, iter):
        save_path = config.save_model_path + "/%07d.tar" % iter
        torch.save({
            "iter": iter + 1,
            "model_dict": self.model.state_dict(),
            "trainer_dict": self.trainer.state_dict()
        }, save_path)

    def setup_train(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        self.trainer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        start_iter = 0
        if self.opt.load_model is not None:
            load_model_path = os.path.join(config.save_model_path, self.opt.load_model)
            checkpoint = torch.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.load_state_dict(checkpoint["model_dict"])
            self.trainer.load_state_dict(checkpoint["trainer_dict"])
            print("Loaded model at " + load_model_path)
        if self.opt.new_lr is not None:
            self.trainer = torch.optim.Adam(self.model.parameters(), lr=self.opt.new_lr)
        return start_iter

    def train_batch_MLE(self, batch):
        ''' Calculate Negative Log Likelihood Loss for the given batch. In order to reduce exposure bias,
                pass the previous generated token as input with a probability of 0.25 instead of ground truth label
        Args:
        :param batch: batch object

        :returns mle_loss
        '''

        step_losses = []
        input_enc, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = batch.get_enc_data()
        input_dec, max_dec_len, dec_lens, target_dec = batch.get_dec_data()

        h_enc, hidden_e = self.model.encoder(input_enc)
        sum_exp_att = None
        prev_h_dec = None
        x_t = get_cuda(torch.LongTensor(len(h_enc)).fill_(self.start_id))
        hidden_d_t = hidden_e
        for t in range(min(max_dec_len, config.max_dec_steps)):

            #use_gound_truth = get_cuda((torch.rand(len(h_enc)) > 0.25)).long()
            #x_t = use_gound_truth * input_dec[:, t] + (1 - use_gound_truth) * x_t
            x_t = input_dec[:, t]

            h_d_t, cell_t = self.model.decoder(x_t, hidden_d_t)
            ct_e, alphat_e, sum_exp_att = self.model.enc_attention(h_d_t, h_enc, enc_padding_mask, sum_exp_att)
            ct_d, prev_h_dec = self.model.dec_attention(h_d_t, prev_h_dec)
            final_dist = self.model.token_gen(h_d_t, ct_e, ct_d, alphat_e, enc_batch_extend_vocab, extra_zeros)

            target = target_dec[:, t]
            log_probs = torch.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.pad_id)
            step_losses.append(step_loss)

            #x_t = torch.multinomial(final_dist, 1).squeeze()
            topv, topi = final_dist.topk(1)
            is_oov = (topi >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * topi.detach() + (is_oov) * self.unk_id  # Replace OOVs with [UNK] token

        losses = torch.sum(torch.stack(step_losses, 1), 1)  # unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        mle_loss = torch.mean(batch_avg_loss)  # Average batch loss
        return mle_loss


    def trainIters(self):
        iter = self.setup_train()
        count = mle_total = 0
        while iter <= config.max_iterations:
            batch = self.batcher.next_batch()
            try:
                mle_loss = self.train_batch_MLE(batch, iter)
            except KeyboardInterrupt:
                print("-------------------Keyboard Interrupt------------------")
                exit(0)

            mle_total += mle_loss
            count += 1
            iter += 1

            if iter % 1000 == 0:
                mle_avg = mle_total / count
                print("iter:", iter, "mle_loss:", "%.3f" % mle_avg)
                count = mle_total = 0

            if iter % 5000 == 0:
                self.save_model(iter)
