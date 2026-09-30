import torch
import torch.nn as nn
import torch.nn.functional as F

class InputLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(InputLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        x_mean = x.pow(2).mean(dim=-1, keepdim=True)
        denom = torch.sqrt(x_mean + self.eps)
        std = x / denom
        rms = std * self.weight
        return rms


class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim) )
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(context_len)

        rotate_angle = position.unsqueeze(-1) * inv_freq.unsqueeze(0)
        self.register_buffer('cos', torch.cos(rotate_angle))
        self.register_buffer('sin', torch.sin(rotate_angle))

    def forward(self, x):
        seq_len = x.size(1)
        x1, x2 = x.chunk(2, dim=-1)

        adjusted_cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
        adjusted_sin = self.sin[seq_len:].unsqueeze(0).unsqueeze(0).to(x.dtype)

        rotation = torch.cat((-x2,x1), dim=-1)
        x_rotate = (x * adjusted_cos) + (adjusted_sin * rotation)
        return x_rotate


class MaskedGQA(nn.Module):
    def __init__(self, embed_dim, context_len, hidden_dim, num_heads, num_kv_groups, head_dim, qk_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups

        assert num_heads % num_kv_groups == 0, "num_kv_groups should be a multiple of num_heads"
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            head_dim = embed_dim // num_heads

        self.head_dim = head_dim
        self.hidden_dim = self.head_dim * num_heads

        self.w_q = nn.Linear(embed_dim, self.hidden_dim, bias=True)
        self.w_k = nn.Linear(embed_dim, self.head_dim * self.num_kv_groups, bias=True)
        self.w_v = nn.Linear(embed_dim, self.head_dim * self.num_kv_groups, bias=True)
        self.w_o = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.rope = RoPE(self.embed_dim, self.context_len)

        if qk_norm:
            self.q_norm = RMSNorm(embed_dim)
            self.k_norm = RMSNorm(embed_dim)
        else:
            self.q_norm = self.k_norm = None

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, num_tokens, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_new = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        k_reshaped = k.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)
        v_reshaped = v.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_reshaped, v_reshaped
            else:
                self.cache_k = torch.cat([self.cache_k, k_reshaped], dim=-1)
                self.cache_v = torch.cat([self.cache_v, v_reshaped], dim=-1)
            keys_base, values_base = self.cache_k, self.cache_v
        else:
            keys_base, values_base = k_reshaped, v_reshaped
            if self.cache_k is not None or self.cache_v is not None:
                self.cache_k, self.cache_v = None, None
                self.ptr_current_pos = 0

        k_new = keys_base.repeat_interleave(self.group_size, dim=1)
        v_new = values_base.repeat_interleave(self.group_size, dim=1)

        attn_score = ( q_new @ k_new.transpose(2, 3) ) / self.head_dim ** 0.5

        num_tokens_q = q_new.shape[-2]
        num_tokens_k = k_new.shape[-2]
        if use_cache:
            q_position = torch.arange(self.ptr_current_pos, self.ptr_current_pos + num_tokens_q, device=q_new.device)
            self.ptr_current_pos += num_tokens_q
        else:
            q_position = torch.arange(num_tokens_q, device=q_new.device)
            self.ptr_current_pos = 0
        k_position = torch.arange(num_tokens_k, device=k_new.device)
        mask = q_position.unsqueeze(-1) < k_position.unsqueeze(-1)

        attn_score = attn_score.masked_fill(mask, -float('inf'))
        attn_weights = F.softmax(attn_score, dim=-1)

        context = torch.bmm(attn_weights, v_new).transpose(1, 2)

        context_vec = context.contiguous().view(b, num_tokens_q, self.hidden_dim)
        output = self.w_o(context_vec)
        return output


class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1+x2


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ff1 = nn.Linear(embed_dim, hidden_dim)
        self.ff2 = nn.Linear(embed_dim, hidden_dim)
        self.ff3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        up = self.ff1(x)
        down = self.ff2(x)

        gate = F.silu(up) * down
        output = self.ff3(gate)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, context_len, hidden_dim, num_heads, num_kv_groups, head_dim, qk_norm):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.masked_gqa = MaskedGQA(embed_dim, context_len, hidden_dim, num_heads, num_kv_groups, head_dim, qk_norm)

        self.residual_connection = ResidualConnection()
        self.norm2 = RMSNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim)

    def forward(self, x, use_cache=False):
        skip1 = x
        x = self.norm1(x)
        x = self.masked_gqa(x, use_cache=use_cache)
        x = self.residual_connection(x, skip1)

        skip2 = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.residual_connection(x, skip2)

        return x


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.output_layer(x)


class Qwen3Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, hidden_dim, num_heads, num_kv_groups, head_dim, qk_norm, num_transformer_blocks):
        super().__init__()
        self.input_layer = InputLayer(vocab_size, embed_dim)
        self.transformer_layer = nn.ModuleList([
            TransformerLayer(embed_dim, context_len, hidden_dim, num_heads, num_kv_groups, head_dim, qk_norm) for _ in range(num_transformer_blocks)
        ])

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x, use_cache=False):
        x = self.input_layer(x)

        for layer in self.transformer_layer:
            x = layer(x, use_cache=use_cache)

        x = self.final_norm(x)
        x = self.output_layer(x)
        return x