from multiprocessing.managers import convert_to_error

import torch
from torch import nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.weight
        return x_norm


class RoPE(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(seq_len)

        rotation_angle = position.unsqueeze(-1) * inv_freq.unsqueeze(0)
        self.cos = torch.cos(rotation_angle)
        self.sin = torch.sin(rotation_angle)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[: self.seq_len, :].unsqueeze(0)
        adj_sin = self.sin[: self.seq_len, :].unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x1 * adj_cos) + (rotation * adj_sin)

        return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads
        self.rope_embed = RoPE(d_model, seq_len)

        self.w_q = nn.Linear(d_model, num_heads * self.head_dim)
        self.w_k = nn.Linear(d_model, num_groups * self.head_dim)
        self.w_v = nn.Linear(d_model, num_groups * self.head_dim)
        self.w_o = nn.Linear(num_heads * self.head_dim, d_model)

    @staticmethod
    def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs= 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs_cis = torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))
        return freqs_cis

    def _repeat_kv(self, x: torch.Tensor, rep: int) -> torch.Tensor:
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(batch, seq_len, n_kv_heads, rep, head_dim)
            .reshape(batch, seq_len, n_kv_heads * rep, head_dim)
        )

    def forward(self, x, mask):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim, bias=False)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim, bias=False)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim, bias=False)

        q = self.rope_embed(q)
        k = self.rope_embed(k)

        k = self._repeat_kv(k, self.group_size)
        v = self._repeat_kv(v, self.group_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.w_o(output)


class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, ):
        super().__init__()
        self.ffl1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffl2 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffl3 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        ffl1_out = self.ffl1(x)
        ffl2_out = self.ffl2(x)
        x = F.silu(ffl1_out) * ffl2_out
        return self.ffl3(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, seq_len):
        super().__init__()
        self.rms_norm_1 = RMSNorm(embed_dim)
        self.grp_attn = GroupedQueryAttention(embed_dim, num_heads, num_groups, seq_len)
        self.sc_1 = SkipConnection()
        self.rms_norm_2 = RMSNorm(embed_dim)
        self.ff = FeedForwardNetwork(embed_dim, embed_dim)
        self.sc_2 = SkipConnection()

    def forward(self, x, mask):
        skip1 = x
        x = self.rms_norm_1(x)
        x = self.grp_attn(x, mask)
        x = self.sc_1(x, skip1)

        skip2 = x
        x = self.rms_norm_2(x)
        x = self.ff(x)
        x = self.sc_2(x, skip2)

        return x


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.output_layer(x)


class LLAMA3Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_groups, seq_len, transformer_layers = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.input_embed = InputEmbedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            *[TransformerBlock(embed_dim, num_heads, num_groups, seq_len) for _ in range(transformer_layers)],
        )
        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, input_idx):
        x = self.input_embed(input_idx)

        num_tokens = input_idx.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        x = self.final_norm(x)
        x = self.output_layer(x)
        return x
