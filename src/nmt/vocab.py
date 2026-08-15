from collections import Counter
from itertools import chain

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = range(len(SPECIAL_TOKENS))


class Vocab:
    def __init__(self, tokens):
        self.itos = SPECIAL_TOKENS + tokens
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, token_lists, min_freq=2):
        counter = Counter(chain.from_iterable(token_lists))
        return cls([tok for tok, count in counter.items() if count >= min_freq])

    def encode(self, tokens, max_length):
        ids = [SOS_IDX] + [self.stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
        if len(ids) < max_length:
            return ids + [PAD_IDX] * (max_length - len(ids))
        return ids[:max_length]

    def decode(self, ids, strip=(PAD_IDX, SOS_IDX, EOS_IDX)):
        return [self.itos[i] if i < len(self.itos) else UNK_TOKEN for i in ids if i not in strip]
