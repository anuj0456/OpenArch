import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.input_embedding(x)

class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(embedding_dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        norm = x / torch.sqrt(rms + self.eps)
        x_norm = norm * self.weight
        return x_norm

class RoPE(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, q_latent_dim, kv_latent_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.head_dim = embedding_dim // num_heads

        self.wq_d = nn.Linear(embedding_dim, q_latent_dim)
        self.w_qk = nn.Linear(q_latent_dim, num_heads * kv_latent_dim)

        self.wkv_d = nn.Linear(embedding_dim, kv_latent_dim)
        self.wv_u = nn.Linear(kv_latent_dim, num_heads * self.head_dim)

        self.w_out = nn.Linear(num_heads * self.head_dim, embedding_dim)

    def forward(self, x):
        b, seq_len, embedding_dim = x.shape
        c_q = self.wq_d(x)
        c_kv = self.wkv_d(x)

        c_qw_qk = self.w_qk(c_q).view(b, seq_len, self.num_heads, self.kv_latent_dim)
        scores = torch.matmul(c_qw_qk.transpose(1, 2), c_kv.transpose(-2, -1)[:, None, ...]) / math.sqrt(self.kv_latent_dim)

        attn_scores = F.softmax(scores, dim=-1)
        v = self.wv_u(c_kv).view(b, seq_len, self.num_heads, -1)

        output = torch.matmul(attn_scores, v.transpose(1, 2)).transpose(1, 2).contiguous()
        output = self.w_out(output.view(b, seq_len, -1))
        return output

class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

class Router(nn.Module):
    def __init__(self):
        super().__init__()

class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

class OutputLayer(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.output_layer(x)