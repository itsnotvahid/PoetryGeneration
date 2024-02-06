import torch
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.register_buffer('mask', torch.tril(torch.ones(1000, 1000)))

    def __rearrange(self, vals):
        batch, sequence, _ = vals.shape
        vals = vals.reshape(batch, sequence, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        return vals

    def forward(self, query, key, value):
        key = self.key(key)
        query = self.query(query)
        value = self.value(value)
        seq = key.shape[1]

        # batch, head, sequence, f
        key = self.__rearrange(key)
        query = self.__rearrange(query)
        value = self.__rearrange(value)
        den = key.shape[-1] ** 0.5
        wgt = query @ key.transpose(-1, -2) / den

        mask = torch.tril(torch.ones((seq, seq), device=next(self.parameters()).device)).type(torch.bool)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions at the beginning
        mask = mask.expand(-1, self.num_heads, -1, -1)  # -1 means not changing that dimension, expand num_heads

        # Now apply the mask
        wgt = wgt.masked_fill(mask == 0, float('-inf'))

        wgt = torch.softmax(wgt, dim=-1)
        z = wgt @ value
        z = z.permute(0, 2, 1, 3)
        z = z.flatten(2)
        output = self.output(z)
        return output
