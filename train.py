import os

from tqdm import tqdm

import torch.nn.functional as F
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

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class Train(object):
    def __init__(self, opt):


        self.opt = opt
        self.dataset = load_dataset('cnn_dailymail', '3.0.0')
        self.vocab = Vocabulary(self.dataset['train'])
        self.start_id = self.vocab[vocab.START_DECODING]
        self.end_id = self.vocab[vocab.STOP_DECODING]
        self.pad_id = self.vocab[vocab.PAD_TOKEN]
        self.unk_id = self.vocab[vocab.UNKNOWN_TOKEN]
        time.sleep(5)

    def save_model(self, iter):
        save_path = config.save_model_path + "/%07d.tar" % iter
        torch.save({
            "iter": iter + 1,
            "model_dict": self.model.state_dict(),
            "trainer_dict": self.trainer.state_dict()
        }, save_path)

    def setup_train(self):
        self.model = Model(self.start_id, self.unk_id, self.pad_id)
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

        wandb.watch(self.model)
        return start_iter

    def train_batch_MLE(self, batch):
        ''' Calculate Negative Log Likelihood Loss for the given batch. In order to reduce exposure bias,
                pass the previous generated token as input with a probability of 0.25 instead of ground truth label
        Args:
        :param batch: batch object

        :returns mle_loss
        '''

        step_losses = []
        enc_data, dec_data = batch
        input_enc, input_enc_length, enc_padding_mask, enc_batch_extend_vocab, extra_zeros = enc_data
        input_dec, dec_lens, target_dec = dec_data
        max_dec_len = dec_lens.max()

        h_enc, hidden_e = self.model.encoder(input_enc)
        sum_exp_att = None
        prev_h_dec = None
        x_t = get_cuda(torch.LongTensor(len(h_enc)).fill_(self.start_id))
        hidden_d_t = hidden_e
        for t in range(min(max_dec_len, config.max_dec_steps)):

            use_gound_truth = get_cuda((torch.rand(len(h_enc)) > 0.25)).long()
            x_t = use_gound_truth * input_dec[:, t] + (1 - use_gound_truth) * x_t
            x_t = input_dec[:, t]

            h_d_t, cell_t = self.model.decoder(x_t, hidden_d_t)
            ct_e, alphat_e, sum_exp_att = self.model.enc_attention(h_d_t, h_enc, enc_padding_mask, sum_exp_att)
            ct_d, prev_h_dec = self.model.dec_attention(h_d_t, prev_h_dec)
            final_dist = self.model.token_gen(h_d_t, ct_e, ct_d, alphat_e, enc_batch_extend_vocab, extra_zeros)

            target = target_dec[:, t]
            log_probs = torch.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.pad_id)
            step_losses.append(step_loss)

            x_t = torch.multinomial(final_dist, 1).squeeze()
            #topv, topi = final_dist.topk(1)
            is_oov = (x_t >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * self.unk_id  # Replace OOVs with [UNK] token

        losses = torch.sum(torch.stack(step_losses, 1), 1)  # unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        mle_loss = torch.mean(batch_avg_loss)  # Average batch loss
        return mle_loss


    def trainIters(self):

        iter = self.setup_train()

        process_batch = lambda batch : collate_batch(batch, self.vocab)

        train_dataset = self.dataset['train']
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=process_batch)
        val_dataset = self.dataset['validation']
        val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=process_batch)

        #history = {}  # Collects per-epoch loss and acc like Keras' fit().
        #history['loss'] = []
        #history['val_loss'] = []
        #history['acc'] = []
        #history['val_acc'] = []

        start = time.time()

        # now we start the main loop
        start_epoch = iter
        for epoch in range(start_epoch, config.max_epochs+1):
            # set models to train mode
            self.model.train()

            # use prefetch_generator and tqdm for iterating through data
            pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
            start_time = time.time()
            loss = 0
            for i, batch in pbar:
                prepare_time = start_time - time.time()
                mle_loss = self.train_batch_MLE(batch)
                process_time = start_time - time.time() - prepare_time

                pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                    process_time / (process_time + prepare_time), epoch, config.max_epochs))
                pbar.set_postfix(train_loss=mle_loss)
                loss += mle_loss
                start_time = time.time()

                wandb.log(mle_loss)

            #history['loss'].append(loss/len(train_data_loader))
            #history['val_loss'].append(val_loss)
            if epoch % 10 == 0:
                self.save_model(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mle', type=str, default="yes")
    parser.add_argument('--train_rl', type=str, default="no")
    parser.add_argument('--mle_weight', type=float, default=1.0)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--new_lr', type=float, default=None)
    opt = parser.parse_args()
    opt.rl_weight = 1 - opt.mle_weight
    print("Training mle: %s, Training rl: %s, mle weight: %.2f, rl weight: %.2f"%(opt.train_mle, opt.train_rl, opt.mle_weight, opt.rl_weight))
    print("intra_encoder:", config.intra_encoder, "intra_decoder:", config.intra_decoder)

    wandb.init(project="nlp-project")
    wandb.config = {"learning_rate": config.lr, "epochs": config.max_epochs, "batch_size": config.batch_size, "hidden_dim" : config.hidden_dim,
                    "emb_dim" : config.emb_dim, "max_enc_setps" : config.max_enc_steps,
                    "max_dec_setps" : config.max_dec_steps, "input_vocab_size" : config.vocab_size, "output_vocab_size" : config.vocab_size}
    train_processor = Train(opt)
    train_processor.trainIters()

