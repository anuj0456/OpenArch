import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size=202000, embedding_dim=5120):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim=5120, eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True)
        rms_norm = torch.sqrt(rms + self.eps)
        x_norm = x / rms_norm

        return x_norm * self.weight


class RoPE(nn.Module):
    def __init__(self, embedding_dim, context_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        positions = torch.arange(context_len)

        rotation_angle = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.cos = torch.cos(rotation_angle)
        self.sin = torch.sin(rotation_angle)

    def forward(self, x):
        x1,x2 = x.chunk(2, dim=-1)
        seq_len = x.size(-2)

        adj_cos = self.cos[: seq_len].unsqueeze(0).unsqueeze(0)
        adj_sin = self.sin[: seq_len].unsqueeze(0).unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)
        x_adjusted = (x * adj_cos) + (rotation * adj_sin)

        return x_adjusted


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, num_kv_groups, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups

        assert num_heads % num_kv_groups == 0 , "num_heads should be divisible by num_kv_groups"
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            head_dim = embed_dim // num_heads

        self.head_dim = head_dim
        self.hidden_dim = self.head_dim * num_heads

        self.w_q = nn.Linear(embed_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)

        self.rope = RoPE(self.head_dim, context_len)

    def forward(self, x, mask=None):
        batch_size, num_tokens, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is None:
            mask = self._calculate_causal_mask(num_tokens, x.device, x.dtype)

        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_prob = F.softmax(attn_score, dim=-1)

        logits = (attn_prob @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.hidden_dim)
        return self.w_o(logits)


class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim=8192):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(embed_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        ll1 = self.linear1(x)
        ll2 = self.linear2(x)

        activation = F.silu(ll1) * ll2
        x = self.linear3(activation)
        return x


class MOE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_experts=128, top_k=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.shared_expert = Expert(embed_dim, hidden_dim)
        self.experts = nn.ModuleList([Expert(embed_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(embed_dim, num_experts)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        hidden_state_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.router(hidden_state_reshaped)

        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(batch_size * seq_len, hidden_dim, device=hidden_state_reshaped.device, dtype=hidden_state_reshaped.dtype)

        unique_experts = torch.unique(top_k_indices)
        for i in unique_experts:
            expert_id = int(i)
            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=1)

            expert_input = hidden_state_reshaped[token_mask]
            expert_weight = top_k_probs[mask].unsqueeze(dim=-1)
            expert_output = self.experts[expert_id](expert_input)
            output[token_mask] += expert_output * expert_weight

        output = output.view(batch_size, seq_len, hidden_dim)
        output = self.shared_expert(output) + output
        return output

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


class MOETransformerBlock(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, num_kv_groups, head_dim, hidden_dim):
        super().__init__()
        self.rmsnorm_1 = RMSNorm(embed_dim)
        self.attention_block = GroupedQueryAttention(embed_dim, context_len, num_heads, num_kv_groups, head_dim)

        self.sc = SkipConnection()
        self.rmsnorm_2 = RMSNorm(embed_dim)
        self.moe_layer = MOE(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        skip1 = x
        x = self.rmsnorm_1(x)

        x = self.attention_block(x, mask)
        x = self.sc(x, skip1)

        skip2 = x
        x = self.rmsnorm_2(x)
        x = self.moe_layer(x)

        x = self.sc(x, skip2)
        return x


class DenseTransformerBlock(nn.Module):
    def __init__(self, embed_dim, context_len, num_heads, num_kv_groups, head_dim, hidden_dim):
        super().__init__()
        self.rmsnorm_1 = RMSNorm(embed_dim)
        self.attention_block = GroupedQueryAttention(embed_dim, context_len, num_heads, num_kv_groups, head_dim)

        self.sc = SkipConnection()
        self.rmsnorm_2 = RMSNorm(embed_dim)
        self.expert = Expert(embed_dim, hidden_dim)

    def forward(self, x, mask= None):
        skip1 = x
        x = self.rmsnorm_1(x)
        x = self.attention_block(x, mask)
        x = self.sc(x, skip1)

        skip2 = x
        x = self.rmsnorm_2(x)
        x = self.expert(x)
        x = self.sc(x, skip2)
        return x


class Llama4MaverickModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, num_heads, num_kv_groups, head_dim, hidden_dim, num_transformer_blocks):
        super().__init__()
        self.input_layer = InputEmbedding(vocab_size, embed_dim)
        blocks = []
        for i in range(num_transformer_blocks):
            cls = DenseTransformerBlock if i % 2 == 0 else MOETransformerBlock
            blocks.append(cls(embed_dim, context_len, num_heads, num_kv_groups, head_dim, hidden_dim))
        self.transformer_blocks = nn.ModuleList(blocks)

        self.final_rms_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def _build_causal_mask(self, query_len, key_len, device, dtype=torch.bool):
        offset = key_len - query_len
        q_pos = torch.arange(query_len, device=device) + offset
        k_pos = torch.arange(key_len, device=device)

        mask = (k_pos[None, :] <= q_pos[:, None]).to(dtype=dtype)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, mask=None):
        x = self.input_layer(x)

        if mask is None:
            seq_len = x.shape[1]
            mask = self._build_causal_mask(seq_len, seq_len, x.device)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        x = self.final_rms_norm(x)
        x = self.output_layer(x)
        return x