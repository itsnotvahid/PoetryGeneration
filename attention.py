import torch
from torch import nn


class MaskedSelfAttention(nn.Module):

    def __init__(self, input_size, attention_dim):
        super().__init__()
        self.key = nn.Linear(input_size, attention_dim)
        self.query = nn.Linear(input_size, attention_dim)
        self.value = nn.Linear(input_size, attention_dim)

        self.register_buffer('mask', torch.tril(torch.ones(1000, 1000)))

    def forward(self, x):
        batch, sequence, embedding_dim = x.size()
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        den = sequence ** 0.5
        num = (query @ key.transpose(-2, -1)) / den

        masked_num = num.masked_fill(self.mask[:sequence, :sequence] == 0, float('-inf'))  # creating Mask
        scaled = torch.nn.functional.softmax(masked_num, dim=-1)
        z = scaled @ value

        return z


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, input_size, dropout):
        super().__init__()

        assert input_size % n_head == 0, 'd_model should be divideable by n_head'
        attention_dim = input_size // n_head
        self.heads = nn.ModuleList([MaskedSelfAttention(input_size, attention_dim) for _ in range(n_head)])
        self.projection = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()

    def forward(self, x):
        y = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.dropout(self.relu(self.projection(y)))
        return y
