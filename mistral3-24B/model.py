import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size=256000, embed_size=5120):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_size)


class RMSNorm(nn.Module):
    def __init__(self, embed_size=5120, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_size))

    def forward(self, x):
        x_mean = x.pow(2).mean(dim=-1, keepdim=True)
        norm = torch.sqrt(x_mean + self.eps)
        rms = x / norm
        x_rms = rms * self.weight
        return x_rms


class RoPE(nn.Module):
    def __init__(self, embed_size=5120, context_len=128000):
        super().__init__()
        self.embed_size = embed_size
        self.context_len = context_len
        N = 10000

        inv_freq = 1. / (N ** (torch.arange(0, embed_size, 2).float() / embed_size))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(context_len)

        rotation_angle = position.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.cos = torch.cos(rotation_angle)
        self.sin = torch.sin(rotation_angle)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[: self.context_len, :].unsqueeze(-1)
        adj_sin = self.sin[: self.context_len, :].unsqueeze(-1)

        rotate = torch.cat((-x2, x1), dim=-1)

        x_rotate = (x * adj_cos) + (rotate * adj_sin)
        return x_rotate


class GroupQueryAttention(nn.Module):
    def __init__(self, embed_size=5120, context_len=128000, num_heads=32, num_kv_groups=8, head_dim=None):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.context_len = context_len

        assert num_heads % num_kv_groups == 0, "num_heads should be a multiple of num_kv_groups"
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert embed_size % num_heads == 0, "embed_size should be a multiple of num_heads"
            head_dim = embed_size // num_heads

        self.head_dim = head_dim
        self.hidden_dim = self.head_dim * self.num_heads

        self.w_q = nn.Linear(self.embed_size, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(self.embed_size, self.num_kv_groups * self.head_dim, bias=False)
        self.w_v = nn.Linear(self.embed_size, self.num_kv_groups * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, self.embed_size, bias=False)

        self.rope = RoPE(self.embed_size, self.context_len)

    def _build_casual_mask(self, context_len, device, dtype=torch.bool):
        mask = torch.tril(torch.ones(context_len, context_len, device=device, dtype=dtype))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, mask=None):
        b, num_tokens, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_score = q @ k.transpose(2,3) / math.sqrt(self.head_dim)

        if mask is None:
            mask = self._build_casual_mask(num_tokens, device=x.device)

        attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attn_prob = F.softmax(attn_score, dim=-1)

        logits = (attn_prob @ v).transpose(1, 2).reshape(b, num_tokens, self.hidden_dim)
        return self.w_o(logits)


class FeedForward(nn.Module):
    def __init__(self, embed_size=5120, hidden_dim=32768):
        super().__init__()
        self.ff1 = nn.Linear(embed_size, hidden_dim)
        self.ff2 = nn.Linear(embed_size, hidden_dim)
        self.ff3 = nn.Linear(hidden_dim, embed_size)

    def forward(self, x):
        ff1_x = self.ff1(x)
        ff2_x = self.ff2(x)

        x_silu = F.silu(ff1_x) * ff2_x
        x = self.ff3(x_silu)
        return x


class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class OutputLayer(nn.Module):
    def __init__(self, embed_size=5120, vocab_size=256000):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return self.output_layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, context_len, hidden_dim, num_heads=40, num_kv_groups=8):
        super().__init__()
        self.embed_size = embed_size
        self.context_len = context_len

        self.pre_norm1 = RMSNorm(embed_size)
        self.gqa_layer = GroupQueryAttention(embed_size, context_len, num_heads, num_kv_groups)

        self.sc = SkipConnection()

        self.pre_norm2 = RMSNorm(embed_size)
        self.ff_layer = FeedForward(embed_size, hidden_dim)

    def forward(self, x, mask=None):
        skip_1 = x
        x = self.pre_norm1(x)
        x = self.gqa_layer(x, mask)

        x = self.sc(x, skip_1)
        skip_2 = x

        x = self.pre_norm2(x)
        x = self.ff_layer(x)
        x = self.sc(x, skip_2)

        return x

class Mistral3Model(nn.Module):
    def __init__(self, vocab_size, embed_size, context_len, hidden_dim,num_heads, num_kv_groups, num_transformer_block=40):
        super().__init__()
        self.num_transformer_block = num_transformer_block

        self.input_layer = InputEmbedding(vocab_size, embed_size)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_size, context_len, hidden_dim, num_heads, num_kv_groups) for _ in range(num_transformer_block)],
        )
        self.final_norm = RMSNorm(embed_size)
        self.output_layer = OutputLayer(embed_size, vocab_size)

    def forward(self, x, mask=None):
        x = self.input_layer(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        x = self.final_norm(x)
        x = self.output_layer(x)
        return x