from torch.utils.data import DataLoader
from keras.preprocessing.sequence import pad_sequences
from torchtext.data.utils import get_tokenizer
from util import config
import torch


tokenizer = get_tokenizer('basic_english')

def get_tokernier():
    return tokenizer

def sentence2ids(sentence, voc, tokenizer, oovs=None):
    oovs_ = oovs if oovs != None else {}

    sentence = tokenizer(sentence)
    ids = [None] * len(sentence)
    ext_ids = [None] * len(sentence)
    for (i, token) in enumerate(sentence):
        w_i = voc[token]
        ids[i] = w_i
        ext_ids[i] = w_i
        if w_i == voc["[UNK]"]:
            if oovs == None:  # Reading an article
                if token not in oovs_:  # adding UNK word to article OOV
                    oovs_[token] = len(oovs_)
                vocab_idx = len(voc) + oovs_[token]  # Map to its temporary article OOV number
            else:  # Reading a resume
                vocab_idx = w_i
                if token in oovs_:  # UNK word in article OOV
                    vocab_idx = len(voc) + oovs_[token]  # Map to its temporary article OOV number

            ext_ids[i] = vocab_idx

    return ids, ext_ids, oovs_

def article_pipeline(seq, voc, tokenizer):
    return sentence2ids(seq, voc, tokenizer)

def resume_pipeline(seq, voc, tokenizer, oovs):
    return sentence2ids(seq, voc, tokenizer, oovs)


def collate_batch(batch, voc_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_len = len(batch)
    inputs_enc, enc_inputs_lengths, encs_padding_mask, encs_batch_extend_vocab, extra_zeros = [None] * batch_len, [
        None] * batch_len, [None] * batch_len, [None] * batch_len, 0
    inputs_dec, dec_inputs_lengths, targets_dec = [None] * batch_len, [None] * batch_len, [None] * batch_len

    for (i, dic) in enumerate(batch):
        article = dic['article']
        resume = dic['highlights']

        input_enc, enc_batch_extend_vocab, oovs = article_pipeline(article, voc_, tokenizer)
        input_dec, target_dec, oovs = resume_pipeline(resume, voc_, tokenizer, oovs)

        enc_inputs_lengths[i] = min(config.max_enc_steps, len(input_enc))
        dec_inputs_lengths[i] = min(config.max_dec_steps, len(input_dec))

        inputs_enc[i] = input_enc
        inputs_dec[i] = input_dec

        enc_batch_extend_vocab.insert(0, voc_['[SOS]'])
        target_dec.insert(0, voc_['[SOS]'])
        enc_batch_extend_vocab.append(voc_['[EOS]'])
        target_dec.append(voc_['[EOS]'])

        encs_batch_extend_vocab[i] = enc_batch_extend_vocab
        targets_dec[i] = target_dec

        extra_zeros = max(extra_zeros, len(oovs))

    inputs_enc = torch.from_numpy(pad_sequences(inputs_enc, maxlen=config.max_enc_steps, value=voc_['[PAD]']))
    encs_batch_extend_vocab = torch.from_numpy(
        pad_sequences(encs_batch_extend_vocab, maxlen=config.max_enc_steps, value=voc_['[PAD]'], dtype='int64'))

    inputs_dec = torch.from_numpy(
        pad_sequences(inputs_dec, maxlen=config.max_dec_steps, value=voc_['[PAD]'], padding='post', truncating='post', dtype='int64'))
    targets_dec = torch.from_numpy(
        pad_sequences(targets_dec, maxlen=config.max_dec_steps, padding='post', truncating='post', value=voc_['[PAD]']))

    enc_inputs_lengths = torch.tensor(enc_inputs_lengths, dtype=torch.int32)
    dec_inputs_lengths = torch.tensor(dec_inputs_lengths, dtype=torch.int32)

    encs_padding_mask = inputs_enc.ne(voc_['[PAD]'])

    extra_zeros = torch.zeros(batch_len, extra_zeros)

    return (inputs_enc.to(device), enc_inputs_lengths.to(device), encs_padding_mask.to(device),
            encs_batch_extend_vocab.to(device), extra_zeros.to(device)), (
           inputs_dec.to(device), dec_inputs_lengths.to(device), targets_dec.to(device))
