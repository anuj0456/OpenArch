import torch
import torch.nn as nn
import math
import numpy as np


class Dropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x)


class InputEmbedding(nn.Module):
    def __init__(self, d_model = 1600, vocab = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.dropout = Dropout(dropout=dropout)

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        i = torch.arange(0, d_model//2)
        div_term = torch.exp(-np.log(10000)  * (2*i//d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1+x2


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * (x + self.bias)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads,seq_len, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "make sure d_model % num_heads == 0"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = Dropout(dropout=dropout)

        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        self.register_buffer('mask', self.mask)

    def mak_attn(self, query, key, value, attn_mask=False):
        _, num_tokens, _ = query.shape

        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attention_score = attention_score.masked_fill(mask_bool == 0, -1e9)

        attention_score = torch.softmax(attention_score, dim=-1)

        if self.dropout is not None:
            attention_score = self.dropout(attention_score)

        return torch.matmul(attention_score, value)

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x = self.mak_attn(query, key, value, attn_mask=mask)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        return self.w_o(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.residual_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.residual_layer(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, dropout, mask, attn_layer_count=25):
        super().__init__()
        self.mask = mask
        self.norm1 = LayerNorm(d_model)

        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, seq_len, dropout)

        self.residual_connection = SkipConnection()
        self.norm2 = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, 4*d_model, dropout)

    def forward(self, x):
        skip_x_1 = self.residual_connection(x)
        x = self.norm1(x)
        x = self.masked_self_attn(x, x, x, self.mask)
        x = self.residual_connection(skip_x_1, x)

        skip_x_2 = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.residual_connection(skip_x_2, x)

        return x


class GPT2Model(nn.Module):
    def __init__(self, d_model, num_heads: int, context_len: int, dropout: float,
                 vocab_size:int, mask: bool, transformer_layers=48):
        super().__init__()
        self.transformer_layers = transformer_layers

        self.input_embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, context_len, dropout)

        self.transformer_block = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, context_len, dropout, mask) for _ in range(transformer_layers)])

        self.layer_norm = LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_block(x)
        x = self.layer_norm(x)
        x_o = self.output_layer(x)
        return x_o



