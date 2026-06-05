import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        norm = torch.sqrt(rms + self.eps)
        x_norm = x / norm

        return x_norm * self.weight


class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        positions = torch.arange(context_len)

        rotation_angle = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.cos = torch.cos(rotation_angle)
        self.sin = torch.sin(rotation_angle)

    def forward(self, x):
        seq_len = x.size(1)
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[: seq_len].unsqueeze(0).unsqueeze(0)
        adj_sin = self.sin[: seq_len].unsqueeze(0).unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)

        x_rotated = (x * adj_cos) + (rotation * adj_sin)

        return x_rotated


class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, num_kv_groups, head_dim, qk_norm = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups

        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            head_dim = embed_dim // num_heads

        self.head_dim = head_dim
        self.hidden_dim = self.head_dim * num_heads

        self.w_q = nn.Linear(embed_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, self.num_kv_groups * self.head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, self.num_kv_groups * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)

        self.rope = RoPE(self.head_dim, context_len)

        if qk_norm:
            self.q_norm = RMSNorm(embed_dim)
            self.k_norm = RMSNorm(embed_dim)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask=None):
        batch_size, num_tokens, embed_dim = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1)

        context = (attn_weight @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.head_dim)

        output = self.w_o(context)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ll1 = nn.Linear(embed_dim, hidden_dim)
        self.ll2 = nn.Linear(embed_dim, hidden_dim)
        self.ll3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        up  = self.ll1(x)
        down = self.ll2(x)

        gate = F.silu(up) * down

        logits = self.ll3(gate)
        return logits


class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.output_layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_len, num_heads, num_kv_groups, head_dim, qk_norm):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_dim)
        self.group_query_attention = GroupQueryAttention(embed_dim, context_len, num_heads, num_kv_groups, head_dim, qk_norm)

        self.sc = SkipConnection()
        self.rms_norm2 = RMSNorm(embed_dim)
        self.ff = FeedForwardNetwork(embed_dim, hidden_dim)

    def _build_causal_mask(self, query_len, key_len, device, dtype=torch.bool):
        offset = key_len - query_len
        q_pos = torch.arange(query_len, device=device) + offset
        k_pos = torch.arange(key_len, device=device)

        mask = (k_pos[None, :] <= q_pos[:, None]).to(dtype=dtype)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, mask=None):
        seq_len = x.shape[1]

        sc1 = x
        x = self.rms_norm1(x)

        if mask is None:
            mask = self._build_causal_mask(seq_len, seq_len, x.device, dtype=x.dtype)

        x = self.group_query_attention(x, mask)
        x = self.sc(x, sc1)

        sc2 = x
        x = self.rms_norm2(x)
        x = self.ff(x)
        x = self.sc(x, sc2)

        return x


class Qwen3Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_len, num_heads, num_kv_groups, head_dim, qk_norm, num_attn_blocks):
        super().__init__()
        self.input_layer = InputLayer(vocab_size, embed_dim)

        self.transformer_blocks = nn.ModuleList(
            *[TransformerBlock(embed_dim, context_len, num_heads, num_kv_groups, head_dim, qk_norm) for _ in range(num_attn_blocks)],
        )

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.input_layer(x)

        for transformer_block in self.transformer_blocks:
            x = self.transformer_blocks(x, mask)

        x = self.final_norm(x)
        output = self.output_layer(x)
        return output
