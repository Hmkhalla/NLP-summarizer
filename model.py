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
        self.input_size = input_size if input_size !=None else config.input_size
        self.emb_dim = emb_dim if emb_dim != None else config.emb_dim
        ### END TO DO ###

        self.embedding = embedding
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim)

    def forward(self, input, hidden):
        ''' Perform word embedding and forward rnn
        :param input: word_id (batch, )
        :param hidden: encoder hidden states (batch, hidden_dim)

        :returns output: word_embedding (batch, embedding_dim)
        :returns hidden : encoder hidden states (batch, hidden_dim)
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden


class IntraTemporalAttention(nn.Module):
    def __init__(self):
        super(IntraTemporalAttention, self).__init__()
        self.W_e = nn.Bilinear(config.hidden_dim, config.hidden_dim, config.attn_dim, bias=False)

    def forward(self, h_enc, h_decoding, enc_padding_mask, sum_exp_att = None):
        ''' Perform INTRA-TEMPORAL ATTENTION ON INPUT SEQUENCE
        :param h_enc: hidden encoding states per word (batch, n_seq, hidden_dim)
        :param h_decoding: hidden decoding states per word (batch, n_seq, hidden_dim)
        :param enc_padding_mask: mask to ignore padded word (empty words for size consideration) (n_batch, n_seq)
        :param sum_exp_att: temporal sum of attention score (batch, n_seq)

        :returns ct_e: context encoding vector (batch, hidden_dim)
        :returns alpha_t: normalized attention score (batch, n_seq)
        :returns sum_exp_att: temporal sum of attention score (batch, n_seq)
        '''
        attn_score = self.W_e(h_decoding, h_enc)

        exp_att = torch.exp(attn_score)
        if sum_exp_att is None:
            sum_exp_att = get_cuda(torch.FloatTensor(exp_att.size()).fill_(1e-10)) + exp_att
        else:
            temp = exp_att
            exp_att = exp_att / sum_exp_att
            sum_exp_att = sum_exp_att + temp

        # assign 0 probability for padded elements
        alpha_t = exp_att * enc_padding_mask
        normalization_factor = alpha_t.sum(1, keepdim=True)
        alpha_t = alpha_t / normalization_factor

        alpha_t = alpha_t.unsqueeze(1)  # bs,1,n_seq
        # Compute encoder context vector
        ct_e = torch.bmm(alpha_t, h_enc)  # bs, 1, 2*n_hid
        ct_e = ct_e.squeeze(1)
        alpha_t = alpha_t.squeeze(1)

        return ct_e, alpha_t, sum_exp_att

class IntraDecoderAttention(nn.Module):
    def __init__(self):
        super(IntraDecoderAttention, self).__init__()
        self.W_d = nn.Bilinear(config.hidden_dim, config.hidden_dim, config.attn_dim, bias=False)

    def forward(self, h_dec, prev_h_dec):
        ''' Perform INTRA-DECODER ATTENTION
        :param h_dec: hidden encoding states per sentence (batch, hidden_dim)
        :param prev_h_dec: previous hidden decoding states per sentence (batch, decoding_step, hidden_dim)

        :returns ct_d: context decoding vector (batch, hidden_dim)
        :returns prev_h_dec: previous hidden decoding states per sentence (batch, decoding_step, hidden_dim)
        '''
        if prev_h_dec is None :
            prev_h_dec = h_dec.unsqueeze(1)
            ct_d = get_cuda(torch.zeros(h_dec.size()))
        else :
            # TO DO find other way than contigous #
            attn = self.W_d(h_dec.unsqueeze(1).expand_as(prev_h_dec).contiguous(), prev_h_dec).squeeze(2)

            alpha_t = F.softmax(attn, dim=1)  # bs, t-1
            ct_d = torch.bmm(alpha_t.unsqueeze(1), prev_h_dec).squeeze(1)  # bs, n_hid
            prev_h_dec = torch.cat([prev_h_dec, h_dec.unsqueeze(1)], dim=1)  # bs, t, n_hid

        return ct_d, prev_h_dec

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.embedding = embedding

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim)

    def forward(self):
        return
