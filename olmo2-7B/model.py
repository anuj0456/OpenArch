import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, input_embed):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_embed)

    def forward(self, x):
        return self.embedding(x)

class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2)).float() / self.embed_dim)
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(self.context_len)

        rotation_angle = position.unsqueeze(1) * inv_freq.unsqueeze(0)

        self.cos = torch.cos(rotation_angle)
        self.sin = torch.sin(rotation_angle)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[:x.size(-2)].unsqueeze(0)
        adj_sin = self.sin[:x.size(-2)].unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)
        adj_x = (x * adj_cos) + (rotation * adj_sin)
        return adj_x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, context_len, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.context_len = context_len
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.w_q = nn.Linear(input_dim, input_dim, bias=False)
        self.w_k = nn.Linear(input_dim, input_dim, bias=False)
        self.w_v = nn.Linear(input_dim, input_dim, bias=False)
        self.w_o = nn.Linear(input_dim, input_dim, bias=False)

        self.qk_norm = RMSNorm(input_dim)
        self.rope = RoPE(input_dim, context_len)

    def forward(self, q, k, v, mask=None):
        b, num_token, _ = q.shape

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = self.qk_norm(query)
        key = self.qk_norm(key)

        query = query.view(b, num_token, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, num_token, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, num_token, self.num_heads, self.head_dim).transpose(1, 2)

        query = self.rope(query)
        key = self.rope(key)

        attention_scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_scores, value).transpose(1, 2).reshape(b, num_token, self.input_dim)
        logits = self.w_o(context)
        return logits


class PostRMSNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.weight
        return x_norm


class SkipConnection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

    def forward(self, x1_input, x2_input):
        return x1_input + x2_input

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        ffl1_out = self.linear1(x)
        ffl2_out = self.linear2(x)

        x = F.silu(ffl1_out) * ffl2_out
        return self.linear3(x)

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, context_len):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sc = SkipConnection(input_dim, output_dim)
        self.multihead_attention = MultiHeadAttention(input_dim, context_len, num_heads=32)
        self.rms_norm1 = RMSNorm(input_dim)

        self.ff = FeedForwardNetwork(input_dim, output_dim)
        self.rms_norm2 = RMSNorm(input_dim)

    def forward(self, x):
        skip1 = x
        x = self.multihead_attention(x, x, x)
        x = self.rms_norm1(x)
        x = self.sc(skip1, x)

        skip2 = x
        x = self.ff(x)
        x = self.rms_norm2(x)
        x = self.sc(skip2, x)

        return x

class OLMO2Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 4096, context_len = 4096, num_attention_block= 32):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.input_embedding = InputEmbedding(vocab_size, embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, embedding_dim, context_len) for _ in range(num_attention_block)],
        )
        self.final_rms_norm = RMSNorm(embedding_dim)
        self.linear_output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.input_embedding(x)

        x = self.transformer_blocks(x)

        x = self.final_rms_norm(x)
        x = self.linear_output(x)
        return x
