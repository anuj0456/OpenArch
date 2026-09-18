import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        x_norm = x / std
        output = self.weight * x_norm
        return output


class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq) , dim=-1)
        position = torch.arange(context_len)

        rotation_angle = position.unsqueeze(-1) * inv_freq.unsqueeze(0)
        self.register_buffer('cos', torch.cos(rotation_angle))
        self.register_buffer('sin', torch.sin(rotation_angle))

    def forward(self, x):
        seq_len = x.size(2)
        x1, x2 = x.chunk(2, dim=-1)

        adjusted_cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
        adjusted_sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)

        rotation = torch.cat((-x2, x1), dim=-1)

        x_rotated = (x * adjusted_cos) + (rotation * adjusted_sin)
        return x_rotated


class GQA(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, head_dim, num_kv_groups):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            head_dim = embed_dim // num_heads

        self.head_dim = head_dim
        self.hidden_dim = head_dim * num_heads

        self.w_q = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.w_k = nn.Linear(self.embed_dim, self.head_dim * num_kv_groups , bias=True)
        self.w_v = nn.Linear(self.embed_dim, self.head_dim * num_kv_groups, bias=True)
        self.w_o = nn.Linear(self.hidden_dim, self.embed_dim, bias=True)

        self.rope = RoPE(head_dim, self.context_len)

        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, n, embed_dim = x.shape

        q = self.w_q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            keys_base, values_base = self.cache_k, self.cache_v
        else:
            keys_base, values_base = k, v
            if self.cache_k is not None or self.cache_v is not None:
                self.cache_k, self.cache_v = None, None
                self.ptr_current_pos = 0

        k = keys_base.repeat_interleave(self.group_size, dim=1)
        v = values_base.repeat_interleave(self.group_size, dim=1)

        attention_score = (q @ k.transpose(2,3)) / self.head_dim**0.5

        num_tokens_q = q.shape[-2]
        num_tokens_k = k.shape[-2]
        device = q.device
        if use_cache:
            q_positions = torch.arange(self.ptr_current_pos,
                                       self.ptr_current_pos + num_tokens_q,
                                       device=device,
                                       dtype=torch.long)
            self.ptr_current_pos += num_tokens_q
        else:
            q_positions = torch.arange(num_tokens_q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_k, device=device, dtype=torch.long)
        attn_mask = q_positions.unsqueeze(-1) <= k_positions.unsqueeze(0)

        attention_score = attention_score.masked_fill(attn_mask == 0, -1e9)
        attn_weight = F.softmax(attention_score, dim=-1)

        contex_vec = (attn_weight @ v).transpose(1, 2)
        contex_vec = contex_vec.contiguous().view(b, n, self.hidden_dim)

        output = self.w_o(contex_vec)
        return output


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ff_1 = nn.Linear(embed_dim, hidden_dim)
        self.ff_2 = nn.Linear(embed_dim, hidden_dim)
        self.ff_3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        up = self.ff_1(x)
        down = self.ff_2(x)

        gate = F.silu(up) * down
        out = self.ff_3(gate)
        return out


class MOE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, top_k, num_experts):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.experts = nn.ModuleList([MLP(embed_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(embed_dim, num_experts)

    def forward(self, hidden_states):
        b, seq_len, embed_dim = hidden_states.shape

        reshape_hidden_states = hidden_states.view(-1, embed_dim)
        router_logits = self.router(reshape_hidden_states)

        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(b * seq_len, embed_dim).to(router_logits.device)
        unique_experts = torch.unique(top_k_indices)

        for expert in unique_experts:
            expert_id = int(expert)
            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=-1)

            expert_input = reshape_hidden_states[token_mask]
            expert_weight = top_k_probs[mask].unsqueeze(-1)
            expert_output = self.experts[expert_id](expert_input)
            output[token_mask] += expert_output * expert_weight

        output = output.view(b, seq_len, embed_dim)
        return output


class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, head_dim, num_kv_groups, hidden_dim, top_k, num_experts):
        super().__init__()
        self.norm_1 = RMSNorm(embed_dim)
        self.attention_layer = GQA(embed_dim, context_len, num_heads, head_dim, num_kv_groups)

        self.residual_connection = ResidualConnection()
        self.shared_mlp = MLP(embed_dim, 32768)

        self.norm_2 = RMSNorm(embed_dim)
        self.moe = MOE(embed_dim, hidden_dim, top_k, num_experts)

    def forward(self, x):
        residual1 = x
        x = self.norm_1(x)
        x = self.attention_layer(x)
        x = self.residual_connection(x, residual1)

        residual2 = x
        x = self.norm_2(x)
        x_moe = self.moe(x)
        x_expert = self.shared_mlp(x)
        x = self.residual_connection(x_moe, x_expert)
        x = self.residual_connection(x, residual2)

        return x


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.output_layer(x)


class GroqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_heads, head_dim, num_kv_groups, hidden_dim, top_k, num_experts, num_trnfmr_blocks):
        super().__init__()
        self.input_layer = InputEmbedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, context_len, num_heads, head_dim, num_kv_groups, hidden_dim, top_k, num_experts)
                for _ in range(num_trnfmr_blocks)])

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x):
        x = self.input_layer(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.final_norm(x)
        output = self.output_layer(x)
        return output
