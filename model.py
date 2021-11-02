import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
import torch.nn.functional as F


from train_util import get_cuda

"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        ## TO_DO ## init weight


        return

    def forward(self, x):
        return
"""

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
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    #def initHidden(self):
    #    return torch.zeros(1, 1, self.hidden_size, device=device)


class IntraTemporalAttention(nn.Module):
    def __init__(self):
        super(IntraTemporalAttention, self).__init__()
        self.W_e = nn.Bilinear(config.hidden_dim, config.hidden_dim, config.attn_dim, bias=False)

    def forward(self, h_enc, h_decoding, enc_padding_mask, sum_exp_att = None):
        attn_score = self.W_e(h_decoding, h_enc)

        exp_att = torch.exp(attn_score)
        if sum_exp_att is None:
            sum_exp_att = get_cuda(torch.FloatTensor(exp_att.size()).fill_(1e-10)) + exp_att
        else:
            temp = exp_att
            exp_att = exp_att / sum_exp_att
            sum_exp_att = sum_exp_att + temp

        # assign 0 probability for padded elements
        alpha = exp_att * enc_padding_mask
        normalization_factor = alpha.sum(1, keepdim=True)
        alpha = alpha / normalization_factor

        alpha = alpha.unsqueeze(1)  # bs,1,n_seq
        # Compute encoder context vector
        ct_e = torch.bmm(alpha, h_enc)  # bs, 1, 2*n_hid
        ct_e = ct_e.squeeze(1)
        alpha = alpha.squeeze(1)

        return ct_e, alpha, sum_exp_att

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.embedding = embedding

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim)

    def forward(self):
        return
