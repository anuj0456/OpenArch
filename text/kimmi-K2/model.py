import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(self, vocab_size=160000, embed_dim=7168):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        norm = torch.sqrt(variance + self.eps)
        x_norm = x / norm

        logits = self.weight * x_norm
        return logits

class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len=128000):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        positions = torch.arange(context_len)

        rotation_angle = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.register_buffer("cos", torch.cos(rotation_angle))
        self.register_buffer("sin", torch.sin(rotation_angle))

    def forward(self, x):
        seq_len = x.size(1)
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[: seq_len].unsqueeze(0).unsqueeze(0)
        adj_sin = self.sin[: seq_len].unsqueeze(0).unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)

        x_rotated = (x * adj_cos) + (adj_sin * rotation)

        return x_rotated

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_len=128000, num_heads=64, qkv_bias=False, latent_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.context_len = context_len
        self.head_dim = hidden_dim // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, hidden_dim // 8)

        self.w_q = nn.Linear(self.embed_dim, self.hidden_dim, bias=qkv_bias)
        self.w_dkv = nn.Linear(self.embed_dim, self.latent_dim, bias=qkv_bias)
        self.w_uk = nn.Linear(self.latent_dim, self.hidden_dim, bias=qkv_bias)
        self.w_uv = nn.Linear(self.latent_dim, self.hidden_dim, bias=qkv_bias)

        self.w_o = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.rope = RoPE(self.head_dim, context_len)

        self.register_buffer("cache_c_kv", None, persistent=False)
        self.ptr_current_pos = 0

    def _reset_cache(self):
        self.cache_c_kv = None
        self.ptr_current_pos = 0

    def _reshape_to_heads(self, x,num_heads, head_dim):
        batch_size, num_toks, _ = x.shape
        return x.view(batch_size, num_toks, num_heads, head_dim).transpose(1, 2).contiguous()

    def forward(self, x, use_cache = False):
        batch_size, num_toks, _ = x.shape

        q = self.w_q(x)
        latent = self.w_dkv(x)

        k = self.w_uk(latent)
        v = self.w_uv(x)

        q = self._reshape_to_heads(q, self.num_heads, self.head_dim)
        k = self._reshape_to_heads(k, self.num_heads, self.head_dim)
        v = self._reshape_to_heads(v, self.num_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        attn_score = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = F.softmax(attn_score / math.sqrt(self.head_dim), dim=-1)

        context = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, num_toks, self.hidden_dim)
        output = self.w_o(context)
        return output


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super().__init__()
        self.ll1 = nn.Linear(embed_dim, hidden_dim)
        self.ll2 = nn.Linear(embed_dim, hidden_dim)
        self.ll3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        gate = self.ll1(x)
        up = self.ll2(x)

        activation = F.silu(gate) * up
        down = self.ll3(activation)
        return down


class MOELayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048, num_experts=384, top_k=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList([MLP(embed_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(self.embed_dim, num_experts)

    def forward(self, x):
        batch_size, num_tokens, hidden_dim = x.shape

        hidden_state_reshaped = x.view(-1, self.embed_dim)
        router_logits = self.router(hidden_state_reshaped)

        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.num_experts, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(batch_size * num_tokens, self.hidden_dim, device=x.device, dtype=x.dtype)

        unique_experts = torch.unique(top_k_indices)
        for expert in unique_experts:
            expert_id = int(expert)
            mask = top_k_indices == expert_id

            token_mask = mask.any(dim=1)
            assert token_mask.any()

            expert_input = hidden_state_reshaped[token_mask]
            expert_weights = top_k_probs[mask].unsqueeze(-1)
            expert_output = self.experts[expert_id](expert_input)

            output[token_mask] += expert_output * expert_weights

        output = output.view(batch_size, num_tokens, self.hidden_dim)
        return output


class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1+x2


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_len, num_heads, qkv_bias, latent_dim, num_experts):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_dim)
        self.mla_layer = MultiHeadLatentAttention(embed_dim, hidden_dim, context_len, num_heads, qkv_bias, latent_dim)

        self.sc = SkipConnection()
        self.rms_norm2 = RMSNorm(embed_dim)
        self.moe = MOELayer(embed_dim, hidden_dim, num_experts)

    def forward(self, x):
        sc1 = x
        x = self.rms_norm1(x)
        x = self.mla_layer(x)
        x = self.sc(x, sc1)

        sc2 = x
        x = self.rms_norm2(x)
        x = self.moe(x)
        x = self.sc(x, sc2)
        return x

class KimmiK2Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_len, num_heads, qkv_bias, latent_dim, num_experts, transformer_blocks):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.input_layer = InputLayer(vocab_size, embed_dim)
        self.dense_ffn = MLP(embed_dim, hidden_dim=18432)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, hidden_dim, context_len, num_heads, qkv_bias, latent_dim, num_experts) for _ in range(transformer_blocks)])

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x):
        x = self.input_layer(x)

        for i, transformer_block in enumerate(self.transformer_blocks):
            if i == 0:
                x = self.dense_ffn(x)
            else:
                x = transformer_block(x)

        x = self.final_norm(x)
        x = self.output_layer(x)
        return x


