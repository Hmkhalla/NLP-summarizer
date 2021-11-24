import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import config, data
import torch.nn.functional as F


from util.train_util import get_cuda



class EncoderRNN(nn.Module):
    def __init__(self, embedding, emb_dim=None, hidden_dim=None):
        super(EncoderRNN, self).__init__()

        self.embedding = embedding
        ### TO DO ###
        ### Make it more robust ###
        self.hidden_dim = hidden_dim if hidden_dim !=None else config.hidden_dim
        self.emb_dim = emb_dim if emb_dim != None else config.emb_dim
        ### END TO DO ###

        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input):
        ''' Perform word embedding and forward rnn
        :param input: word_id in sequence (batch_size, max_enc_steps)
        :param embedded: word_vectors (batch_size, max_enc_steps, emb_dim)

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
            attn_score[:, i] = self.W_e(h_d_t, h_enc[:, i, :]).view(-1)

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
                attn_score[:, i] = self.W_d(h_d_t, prev_h_dec[:, i, :]).view(-1)

            alpha_t = F.softmax(attn_score, dim=1)  # bs, t-1
            ct_d = torch.bmm(alpha_t.unsqueeze(1), prev_h_dec).squeeze(1)  # bs, n_hid
            prev_h_dec = torch.cat([prev_h_dec, h_d_t.unsqueeze(1)], dim=1)  # bs, t, n_hid

        return ct_d, prev_h_dec

class TokenGeneration(nn.Module):
    def __init__(self):
        super(TokenGeneration, self).__init__()
        # TO DO share weigths #
        self.lin_out = nn.Linear(3*2*config.hidden_dim, config.vocab_size)
        self.lin_u = nn.Linear(3*2*config.hidden_dim, 1)


    def forward(self, h_d_t, ct_e, ct_d, alphat_e, enc_batch_extend_vocab, extra_zeros):
        ''' Perform TOKEN GENERATION AND POINTER
        :param h_d_t: decoder hidden state at current time step (batch_size, 2*hidden_dim)
        :param ct_e: encoder context vector for decoding_step (eq 5 in https://arxiv.org/pdf/1705.04304.pdf) (batch_size, 2*hidden_dim)
        :param ct_d: decoder context vector for decoding_step (batch_size, 2*hidden_dim)
        :param alphat_e: normalized encoder attention score (batch_size, max_enc_steps)
        :param enc_batch_extend_vocab: Input batch that stores word ids including OOVs meaning going
        from 0 to vocab_size+n for n OOVs(batch_size, max_enc_steps)

        :returns final_dist: final output distribution including OOV (batch_size, vocab_size + max OOV_nb)
        '''

        hidden_states = torch.cat([h_d_t, ct_e, ct_d], dim=1)
        p_u = torch.sigmoid(self.lin_u(hidden_states)) # bs,1

        vocab_dist = F.softmax(self.lin_out(hidden_states), dim=1)
        vocab_dist = p_u * vocab_dist

        attn_dist = alphat_e
        attn_dist = (1 - p_u) * attn_dist

        # pointer mechanism (as suggested in eq 9 https://arxiv.org/pdf/1704.04368.pdf)
        if extra_zeros is not None:
            vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist)

        return final_dist


class DecoderRNN(nn.Module):
    def __init__(self, embedding):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTMCell(config.emb_dim, 2*config.hidden_dim)

    def forward(self, input, h_enc):
        embedded = self.embedding(input)
        h_d_t, cell_t = self.lstm(embedded, h_enc)
        return h_d_t, cell_t

class Model(nn.Module):
    def __init__(self, start_id, unk_id, pad_id):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.encoder = EncoderRNN(self.embedding)
        self.decoder = DecoderRNN(self.embedding)
        self.enc_attention = IntraTemporalAttention()
        self.dec_attention = IntraDecoderAttention()
        self.token_gen = TokenGeneration()

        self.start_id = start_id
        self.unk_id = unk_id
        self.pad_id = pad_id

    def forward(self, input):
        input_embedded = self.embedding(input)
        h_enc, hidden_e = self.encoder(input_embedded)
        output_embedded = self.embedding(self.start_id)
        hidden_d = hidden_e
        enc_padding_mask = get_cuda(torch.ones_like(input_embedded))
        enc_padding_mask[input_embedded==self.pad_id] = 0
        sum_exp_att = None
        prev_h_dec = None
        resume = torch.zeros(config.batch_size, config.max_dec_steps)
        for t in range(config.max_dec_steps):
            h_d_t, cell_d_t = self.decoder(output_embedded, hidden_d)
            ct_e, alphat_e, sum_exp_att = self.enc_attention(h_d_t, h_enc, enc_padding_mask, sum_exp_att)
            ct_d, prev_h_dec = self.dec_attention(h_d_t, prev_h_dec)
            final_dist = self.token_gen(h_d_t, ct_e, ct_d, alphat_e, input_embedded)
            # TO DO ID generation #
            # resume[:, t] = ...
        return resume
