import math
import torch
from torch import nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(self, vocab_size, input_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        x_norm = x / rms

        logits = x_norm * self.weight
        return logits


class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = (1. / N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        positions = torch.arange(context_len)

        rot_angle = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.register_buffer('cos', torch.cos(rot_angle))
        self.register_buffer('sin', torch.sin(rot_angle))

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[: seq_len ].unsqueeze(0).unsqueeze(0)
        adj_sin = self.sin[: seq_len ].unsqueeze(0).unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)

        x_adjusted = (x * adj_cos) + (rotation * adj_sin)

        return x_adjusted


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, context_len, n_heads, head_dim, num_kv_groups, forward_token_count=0, qk_norm = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.n_heads = n_heads
        self.num_kv_groups = num_kv_groups
        self.forward_token_count = forward_token_count

        assert n_heads % num_kv_groups == 0, "num attention heads should be divisible by num_kv_groups"
        self.group_size = n_heads // num_kv_groups

        if head_dim is None:
            head_dim = embed_dim // n_heads

        self.head_dim = head_dim
        self.hidden_dim = head_dim * n_heads

        if qk_norm:
            self.q_norm = RMSNorm(embed_dim)
            self.k_norm = RMSNorm(embed_dim)
        else:
            self.q_norm = self.k_norm = None

        self.rope = RoPE(embed_dim, context_len)

        self.w_q = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(self.embed_dim, self.num_kv_groups * self.head_dim, bias=False)
        self.w_v = nn.Linear(self.embed_dim, self.num_kv_groups * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        att_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            att_score = att_score.masked_fill(mask == 0, -1e9)
        att_weight = F.softmax(att_score, dim=-1)

        context = (att_weight @ v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        output = self.w_o(context)
        return output


class MultiTokenPredictionHead(nn.Module):
    def __init__(self, embed_dim, vocab_size, transformer_block_fn):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = RMSNorm(embed_dim)
        self.transformer_layer = transformer_block_fn()
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    @staticmethod
    def mtp_loss_fn(input_ids, labels=None, mtp_logits_list=None, mtp_lamda=0.3):
        batch_size, seq_len, embed_dim = input_ids.size()
        loss = F.cross_entropy(input_ids[:, :-1].reshape(-1, embed_dim), labels[:, 1:].reshape(-1), ignore_index=0)
        for depth, aux_logits in enumerate(mtp_logits_list):
            offset = depth + 2
            if offset >= seq_len:
                break
            aux_loss = F.cross_entropy(aux_logits[:, :-offset].reshape(-1, embed_dim), labels[:, offset:].reshape(-1), ignore_index=0)
            loss = loss + mtp_lamda * aux_loss
        return input_ids, loss

    def forward(self, h_prev, future_token_emb, mask=None):
        combined = self.norm(h_prev) + future_token_emb
        h_next = self.transformer_layer(combined, mask=mask)
        logits = self.lm_head(h_next)
        return h_next, logits


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_layer1 = nn.Linear(input_dim, hidden_dim)
        self.linear_layer2 = nn.Linear(input_dim, hidden_dim)
        self.linear_layer3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        up = self.linear_layer1(x)
        gate = F.silu(self.linear_layer1(x))

        down = up * gate
        x = self.linear_layer3(down)
        return x


class MOE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_experts, top_k):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([FFN(embed_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(embed_dim, num_experts)

    def forward(self, hidden_state, mask=None):
        batch_size, seq_len, _ = hidden_state.size()

        hidden_state_reshaped = hidden_state.view(-1, self.embed_dim)
        router_logits = self.router(hidden_state_reshaped)

        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_prob = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(batch_size * seq_len, self.embed_dim, device=hidden_state.device, dtype=hidden_state.dtype)

        unique_experts = torch.unique(top_k_indices)
        for expert in unique_experts:
            expert_id = int(expert)
            mask = (top_k_indices == expert_id)

            token_mask = mask.any(dim=-1)
            expert_input = hidden_state_reshaped[token_mask]
            expert_weight = top_k_prob[mask].unsqueeze(-1)
            expert_output = self.experts[expert_id](expert_input)
            output[token_mask] += expert_output * expert_weight

        output = output.view(batch_size, seq_len, self.embed_dim)
        return output


class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class OutputLayer(nn.Module):
    def __init__(self, output_dim, vocab_size):
        super().__init__()
        self.output_dim = output_dim
        self.output_layer = nn.Linear(output_dim, vocab_size)

    def forward(self, x):
        return self.output_layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_len, n_heads, head_dim, num_kv_groups, forward_token_count, qk_norm, num_experts, top_k):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_dim)
        self.gqa = GroupedQueryAttention(embed_dim, context_len, n_heads, head_dim, num_kv_groups, forward_token_count, qk_norm)

        self.dense_ffn = FFN(embed_dim, hidden_dim=12288)

        self.sc = SkipConnection()
        self.rms_norm2 = RMSNorm(embed_dim)
        self.moe = MOE(embed_dim, hidden_dim, num_experts, top_k)

    def forward(self, x, mask=None, use_dense=False):
        sc1 = x
        x = self.rms_norm1(x)
        x = self.gqa(x, mask)
        x = self.sc(sc1, x)

        sc2 = x
        x = self.rms_norm2(x)
        x = self.dense_ffn(x) if use_dense else self.moe(x)
        x = self.sc(sc2, x)

        return x


class GLM45Model(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim,
                 hidden_dim,
                 context_len,
                 n_heads,
                 head_dim,
                 num_kv_groups,
                 forward_token_count,
                 qk_norm,
                 num_experts,
                 top_k,
                 num_transformer_blocks,
                 mtp_depth,
                 mtp_lambda=0.3):
        super().__init__()

        self.mtp_lamda = mtp_lambda

        self.input_layer = InputLayer(vocab_size, embed_dim)

        def transformer_fn():
            return TransformerBlock(embed_dim, hidden_dim, context_len, n_heads, head_dim, num_kv_groups, forward_token_count, qk_norm, num_experts, top_k)

        self.transformer_layer = nn.ModuleList([transformer_fn() for _ in range(num_transformer_blocks)])

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

        self.mtp_heads = nn.ModuleList([
            MultiTokenPredictionHead(embed_dim, hidden_dim, transformer_fn) for _ in range(mtp_depth)
        ]) if mtp_depth > 0  else None


    def forward(self, x, mask=None, labels=None):
        x = self.input_layer(x)

        for i, layer in enumerate(self.transformer_layer):
            if i < 3:
                x = layer(x, mask, True)
            else:
                x = layer(x, mask)

        x_norm = self.final_norm(x)
        x = self.output_layer(x_norm)

        mtp_logits_list = []
        if self.mtp_heads and labels is not None:
            h = x_norm
            for depth, heads in enumerate(self.mtp_heads):
                shift_id = F.pad(x[:, depth + 1:], (0, depth + 1), value=0)
                future_emb = self.input_layer(shift_id)
                h, logits = heads(h, future_emb, mask=mask)
                mtp_logits_list.append(logits)

        x, loss = MultiTokenPredictionHead.mtp_loss_fn(x, labels, mtp_logits_list, self.mtp_lamda)

        return x, loss