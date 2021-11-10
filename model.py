import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
import torch.nn.functional as F


from train_util import get_cuda


embedding = nn.Embedding(config.vocab_size, config.emb_dim)

class EncoderRNN(nn.Module):
    def __init__(self, input_size=None, emb_dim=None, hidden_dim=None):
        super(EncoderRNN, self).__init__()
        ### TO DO ###
        ### Make it more robust ###
        self.hidden_dim = hidden_dim if hidden_dim !=None else config.hidden_dim
        self.emb_dim = emb_dim if emb_dim != None else config.emb_dim
        ### END TO DO ###

        self.embedding = embedding
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input, hidden):
        ''' Perform word embedding and forward rnn
        :param input: word_id in sequence (batch_size, max_enc_steps)

        :returns h_enc_seq: hidden encoding states for all sentence (batch_size, max_enc_steps, 2*hidden_dim)
        :returns hidden : Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, 2*hidden_dim)
        '''
        embedded = self.embedding(input)
        h_enc_seq, hidden = self.lstm(embedded)

        h, c = hidden  # shape of h: 2, bs, n_hid
        h = torch.cat(list(h), dim=1)  # bs, 2*n_hid
        c = torch.cat(list(c), dim=1)
        return h_enc_seq, (h, c)


class IntraTemporalAttention(nn.Module):
    def __init__(self):
        super(IntraTemporalAttention, self).__init__()
        self.W_e = nn.Bilinear(2*config.hidden_dim, 2*config.hidden_dim, config.attn_dim, bias=False)

    def forward(self, h_d_t, h_enc, enc_padding_mask, sum_exp_att = None):
        ''' Perform INTRA-TEMPORAL ATTENTION ON INPUT SEQUENCE
        :param h_d_t: decoder hidden state at current time step (batch, 2*hidden_dim)
        :param h_enc: hidden encoding states for all sentence (batch_size, max_enc_steps, 2*hidden_dim)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, max_enc_steps) with values of 0 for pad tokens & 1 for others
        :param sum_exp_att: summation of attention weights from previous decoder time steps (batch_size, max_enc_steps)

        :returns ct_e: encoder context vector for decoding_step (eq 5 in https://arxiv.org/pdf/1705.04304.pdf) (batch_size, 2*hidden_dim)
        :returns alphat_e: normalized encoder attention score (batch_size, max_enc_steps)
        :returns sum_exp_att: actualised summation of attention weights from decoder time steps (batch_size, max_enc_steps)
        '''

        # attn_score = self.W_e(h_d_t.unsqueeze(1).expand_as(h_enc), h_enc)
        attn_score = get_cuda(torch.zeros((config.batch_size, config.max_enc_steps), dtype=torch.float))
        for i in range(config.max_enc_steps):
            attn_score[i] = self.W_e(h_d_t, h_enc[:, i, :])

        exp_att = torch.exp(attn_score)
        if sum_exp_att is None:
            sum_exp_att = get_cuda(torch.FloatTensor(exp_att.size()).fill_(1e-10)) + exp_att
        else:
            temp = exp_att
            exp_att = exp_att / sum_exp_att
            sum_exp_att = sum_exp_att + temp

        # assign 0 probability for padded elements
        alphat_e = exp_att * enc_padding_mask
        normalization_factor = alphat_e.sum(1, keepdim=True)
        alphat_e = alphat_e / normalization_factor

        alphat_e = alphat_e.unsqueeze(1)  # bs,1,n_seq
        # Compute encoder context vector
        ct_e = torch.bmm(alphat_e, h_enc)  # bs, 1, 2*n_hid
        ct_e = ct_e.squeeze(1)
        alphat_e = alphat_e.squeeze(1)

        return ct_e, alphat_e, sum_exp_att

class IntraDecoderAttention(nn.Module):
    def __init__(self):
        super(IntraDecoderAttention, self).__init__()
        self.W_d = nn.Bilinear(2*config.hidden_dim, 2*config.hidden_dim, config.attn_dim, bias=False)

    def forward(self, h_d_t, prev_h_dec):
        ''' Perform INTRA-DECODER ATTENTION
        :param h_d_t: decoder hidden state at current time step (batch_size, 2*hidden_dim)
        :param prev_h_dec: previous hidden decoding states (batch_size, decoding_step-1, 2*hidden_dim)

        :returns ct_d: decoder context vector for decoding_step (batch_size, 2*hidden_dim)
        :returns prev_h_dec: previous hidden decoding states (batch_size, decoding_step, 2*hidden_dim)
        '''
        if prev_h_dec is None :
            prev_h_dec = h_d_t.unsqueeze(1)
            ct_d = get_cuda(torch.zeros(h_d_t.size()))
        else :
            # TO DO find other way than contigous #
            #attn_score = self.W_d(h_d_t.unsqueeze(1).expand_as(prev_h_dec).contiguous(), prev_h_dec).squeeze(2)
            attn_score = get_cuda(torch.zeros(prev_h_dec.size()[:-1], dtype=torch.float))
            for i in range(attn_score.size()[-1]):
                attn_score[i] = self.W_e(h_d_t, prev_h_dec[:, i, :])

            alpha_t = F.softmax(attn_score, dim=1)  # bs, t-1
            ct_d = torch.bmm(alpha_t.unsqueeze(1), prev_h_dec).squeeze(1)  # bs, n_hid
            prev_h_dec = torch.cat([prev_h_dec, h_d_t.unsqueeze(1)], dim=1)  # bs, t, n_hid

        return ct_d, prev_h_dec

class TokenGeneration(nn.Module):
    def __init__(self):
        super(TokenGeneration, self).__init__()
        # TO DO share weigths #
        self.lin_out = nn.Linear(3*config.hidden_dim, config.vocab_size)
        self.lin_u = nn.Linear(3*config.hidden_dim, 1)


    def forward(self, h_dec, ct_e, ct_d, enc_batch_extend_vocab):
        hidden_states = torch.cat([h_dec, ct_e, ct_d], 1)
        p_u = torch.sigmoid(self.lin_u(hidden_states)) # bs,1


        vocab_dist = F.softmax(self.lin_out(hidden_states), dim=1)
        vocab_dist = p_u * vocab_dist

        attn_dist = ct_e
        attn_dist = (1 - p_u) * attn_dist

        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist)

        return final_dist


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.embedding = embedding

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim)

    def forward(self):
        return
