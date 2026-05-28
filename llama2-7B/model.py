import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm = x / rms * self.weight
        return norm


class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** torch.arange(0, embed_dim, 2).float() / embed_dim)
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(context_len)

        rot_angle = position.unsqueeze(1) * inv_freq.unsqueeze(0)

        self.cos = torch.cos(rot_angle)
        self.sin = torch.sin(rot_angle)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[: self.context_len, :].unsqueeze(0)
        adj_sin = self.sin[: self.context_len, :].unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * adj_cos) + (rotation * adj_sin)

        return x_rotated.to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_out = nn.Linear(embed_dim, embed_dim)

        self.rope_layer = RoPE(embed_dim, seq_len)

        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        self.register_buffer('mask', self.mask)

    def forward(self, q, k, v, mask=None):
        b, num_tokens, embed_dim = q.shape

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        query = self.rope_layer(query)
        key = self.rope_layer(key)

        attention_score = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(embed_dim)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        else:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attention_score = attention_score.masked_fill(mask_bool, -1e9)

        attention_score = F.softmax(attention_score / math.sqrt(embed_dim), dim=-1)

        output_vec = (attention_score * value).transpose(1, 2)
        output_vec = output_vec.reshape(b, num_tokens, self.embed_dim)

        logits = self.w_out(output_vec)
        return logits


class SkipAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_x1, input_x2):
        return input_x1 + input_x2


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(embed_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        ff1_x = self.linear1(x)
        ff2_x = self.linear2(x)

        activation = F.silu(ff1_x) * ff2_x
        out = self.linear3(activation)
        return out


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_dim)
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads, seq_len)
        self.sc = SkipAttention()

        self.rms_norm2 = RMSNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, embed_dim)

    def forward(self, x):
        skip1 = x
        x = self.rms_norm1(x)
        x = self.multihead_attn(x)
        x = self.sc(x, skip1)

        skip2 = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = self.sc(x, skip2)

        return x


class Llama2Model(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, vocab_size, attention_blocks):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.attention_blocks = attention_blocks
        self.vocab_size = vocab_size

        self.input_embedding = InputEmbedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, seq_len) for _ in range(attention_blocks)])
        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x):
        x = self.input_embedding(x)

        for attention_block in self.attention_blocks:
            x = attention_block(x)

        x = self.final_norm(x)
        x = self.output_layer(x)
        return x