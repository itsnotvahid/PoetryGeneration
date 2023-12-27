from torch import nn

from attention import MultiHeadAttention
from sub_model import ResidualConnection


class FeedForward(nn.Module):
    def __init__(self, inp, dropout):
        super().__init__()
        self.layer = nn.Linear(inp, inp * 4)
        self.layer2 = nn.Linear(inp * 4, inp)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.layer(x))
        x = self.dropout(x)
        y = self.layer2(x)
        return self.dropout(self.relu(y))


class DecoderBlock(nn.Module):

    def __init__(self, input_size, num_heads, dropout):
        super().__init__()
        self.mha = ResidualConnection(MultiHeadAttention(num_heads, input_size, dropout), input_size)
        self.ff = ResidualConnection(FeedForward(input_size, dropout), input_size)

    def forward(self, x):
        y = self.mha(x)
        y = self.ff(y)
        return y
