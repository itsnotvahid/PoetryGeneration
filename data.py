import re

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

from configs import configs

with open('poet.txt') as f:
    text = f.read()


def clean(corpus):
    new_text = list()
    for i in corpus:
        if re.match('[ا-ی|\s]', i):
            new_text.append(i)
    corpus = ''.join(new_text).splitlines()
    return ''.join(corpus)


text = clean(text)
vocab = build_vocab_from_iterator(text)
index2char = vocab.get_itos()

train, valid = text[:int(0.9*len(text))], text[int(0.9*len(text)):]


class MyDataset(Dataset):

    def __init__(self, data, seq, step):
        char_map = torch.LongTensor([vocab[c] for c in data])
        char_map = char_map.unfold(0, seq, step)
        self.x, self.y = (lambda x: (x[:, :-1], x[:, 1:]))(char_map)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


valid_set = MyDataset(valid, configs['unfold'], configs['step'])
train_set = MyDataset(train, configs['unfold'], configs['step'])

train_loader = DataLoader(train_set, shuffle=True, batch_size=configs['batch_size'])
valid_loader = DataLoader(valid_set, shuffle=True, batch_size=configs['batch_size'])
