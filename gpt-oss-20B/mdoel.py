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
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        x_norm = x / std
        output = x_norm * self.weight
        return output


class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        positions = torch.arange(context_len)

        rotation_angle = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        self.register_buffer('cos', torch.cos(rotation_angle))
        self.register_buffer('sin', torch.sin(rotation_angle))

    def forward(self, x):
        seq_len = x.size(2)
        x1, x2 = x.chunk(2, dim=-1)

        adjusted_cos = self.cos[: seq_len].unsqueeze(0).unsqueeze(0)
        adjusted_sin = self.sin[: seq_len].unsqueeze(0).unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)

        x_rotated = (x * adjusted_cos) + (rotation * adjusted_sin)
        return x_rotated


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, head_dim, num_kv_groups):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups

        assert num_heads % num_kv_groups == 0, "num_kv_groups must be multiple of num_heads"
        self.group_size = num_heads // num_kv_groups

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if head_dim is None:
            head_dim = embed_dim // num_heads

        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim

        self.w_q = nn.Linear(embed_dim, self.hidden_dim, bias=True)
        self.w_k = nn.Linear(embed_dim, head_dim * num_kv_groups, bias=True)
        self.w_v = nn.Linear(embed_dim, head_dim * num_kv_groups,  bias=True)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=True)

        self.rope = RoPE(head_dim, context_len)

    def _get_causal_mask(self, context_len, device, dtype=torch.bool):
        mask = torch.tril(torch.ones(context_len, context_len, device=device, dtype=dtype))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, mask=None):
        b, num_tokens, embed_dim = x.shape

        q = self.w_q(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is None:
            mask = self._get_causal_mask(attention_score)

        attention_score = attention_score.masked_fill(mask == 0, -1e9)
        sinks = self.sinks.view(1, self.num_heads, 1, 1).expand(b, -1, num_tokens, 1)
        combined = torch.cat([attention_score, sinks], dim=-1)
        attention_weight = F.softmax(combined, dim=-1)[..., :-1]


        context = (attention_weight @ v).transpose(1, 2).reshape(b, num_tokens, self.hidden_dim)
        logits = self.w_o(context)
        return logits


class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ff1 = nn.Linear(embed_dim, hidden_dim)
        self.ff2 = nn.Linear(embed_dim, hidden_dim)
        self.ff3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        up = self.ff1(x)
        down = self.ff2(x)

        gate = F.silu(up) * down

        logits = self.ff3(gate)
        return logits


class MOE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, top_k, num_experts):
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList([MLP(embed_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(embed_dim, num_experts)

    def forward(self, hidden_state):
        b, seq_len, embed_dim = hidden_state.shape

        reshaped_hidden_state = hidden_state.view(-1, embed_dim)
        router_logits = self.router(reshaped_hidden_state)

        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(b * seq_len, embed_dim, device=reshaped_hidden_state.device, dtype=reshaped_hidden_state.dtype)
        unique_experts = torch.unique(top_k_indices)

        for expert in unique_experts:
            expert_id = int(expert)
            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=-1)

            expert_input = reshaped_hidden_state[token_mask]
            expert_weight = top_k_probs[mask].unsqueeze(-1)
            expert_output = self.experts[expert_id](expert_input)
            output[token_mask] += expert_output * expert_weight

        output = output.view(b, seq_len, embed_dim)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, head_dim, num_kv_groups, hidden_dim, top_k, num_experts):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_dim)
        self.attn_block = GroupedQueryAttention(embed_dim, context_len, num_heads, head_dim, num_kv_groups)

        self.rc = ResidualConnection()

        self.rms_norm2 = RMSNorm(embed_dim)
        self.moe = MOE(embed_dim, hidden_dim, top_k, num_experts)

    def forward(self, x):
        residual1 = x
        x = self.rms_norm1(x)
        x = self.attn_block(x)
        x = self.rc(x, residual1)

        residual2 = x
        x = self.rms_norm2(x)
        x = self.moe(x)
        x = self.rc(x, residual2)

        return x


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.output_layer(x)


class GPTOSSModel(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim,
                 context_len,
                 num_heads,
                 head_dim,
                 num_kv_groups,
                 hidden_dim,
                 top_k,
                 num_experts,
                 num_attention_blocks):
        super().__init__()
        self.input_layer = InputEmbedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, context_len, num_heads, head_dim, num_kv_groups, hidden_dim, top_k, num_experts) for _ in range(num_attention_blocks)])

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.output_layer(x)
        return x

