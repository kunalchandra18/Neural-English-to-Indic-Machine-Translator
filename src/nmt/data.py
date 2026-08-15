import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def encode_corpus(vocab, token_lists, seq_length):
    return np.array([vocab.encode(tokens, seq_length) for tokens in token_lists])


def make_dataset(*arrays):
    return TensorDataset(*[torch.from_numpy(a).long() for a in arrays])


def make_loader(*arrays, batch_size, shuffle=False, drop_last=False):
    return DataLoader(make_dataset(*arrays), batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)
