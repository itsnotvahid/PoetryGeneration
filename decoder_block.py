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
        self.ln_1 = nn.LayerNorm(input_size)
        self.ln_2 = nn.LayerNorm(input_size)
        self.mha = MultiHeadAttention(num_heads, input_size)
        self.ff = FeedForward(input_size, dropout)

    def forward(self, x):
        temp = x
        y = self.mha(x, x, x)
        x = self.ln_1(temp + y)

        temp = x
        y = self.ff(x)
        y = self.ln_2(temp + y)

        return y
