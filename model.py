import torch
from torch import nn
from data import vocab
from decoder_block import DecoderBlock
from sub_model import PositionalEncoding


class GPTLanguageModel(nn.Module):

    def __init__(self, d_model, num_heads, n_block, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, dropout) for _ in range(n_block)])
        self.embedding = nn.Embedding(len(vocab), d_model)
        self.dropout = nn.Dropout()
        self.position_encoding = PositionalEncoding(d_model)
        self.lm_head = nn.Linear(d_model, len(vocab))

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.position_encoding(x)
        for block in self.blocks:
            x = block(x)
        y = self.lm_head(self.dropout(x))
        return y