import os.path
import errno
from collections import OrderedDict
from tqdm import tqdm

from torchtext.data.utils import get_tokenizer

import torchtext
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.vocab.vocab import Vocab



def yield_tokens(data_iter):
  tokenizer = get_tokenizer('basic_english')
  for data in tqdm(data_iter, desc=f"Tokenizing"):
    article = data['document']
    resume = data['summary']
    yield tokenizer(article)
    yield tokenizer(resume)


START_DECODING = "[SOS]"
STOP_DECODING =  "[EOS]"
PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"

class Vocabulary(Vocab):
    def __init__(self, dataset, yield_token=yield_tokens, vocab_size=150000, save_path="data/vocab"):
      super(Vocabulary, self).__init__(None)
      if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
      if os.path.exists(save_path):
        print(f'Loading vocabulary from {save_path} .......')
        with open(save_path, "r") as f:
          tokens = f.read().splitlines()[:vocab_size]
        print('Succesfuly loaded')
      else:
        voc = build_vocab_from_iterator(yield_tokens(dataset), specials=[START_DECODING, STOP_DECODING, PAD_TOKEN, UNKNOWN_TOKEN], special_first=True)
        tokens = voc.get_itos()[:vocab_size]
        with open(save_path, "w") as f:
          for token in tqdm(tokens, desc=f"saving vocabulary in {save_path}"):
              f.write(token + "\n")  
      voc = vocab(OrderedDict([(token, 1) for token in tokens]))
      voc.set_default_index(voc['[UNK]'])
      self.vocab = voc.vocab
