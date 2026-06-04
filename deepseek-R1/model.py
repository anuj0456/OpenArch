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
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        norm = x / torch.sqrt(rms + self.eps)
        x_norm = norm * self.weight
        return x_norm


class RoPE(nn.Module):
    def __init__(self, embed_dim, cotext_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.cotext_len = cotext_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(cotext_len)

        rot_angle = position.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.cos = torch.cos(rot_angle)
        self.sin = torch.sin(rot_angle)

    def forward(self, x):
        x1,x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[: self.context_len, :].unsqueeze(0)
        adj_sin = self.sin[: self.context_len, :].unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * adj_cos) + (adj_sin * rotation)

        return x_rotated


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embedding_dim, context_len, num_heads, q_latent_dim, kv_latent_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.head_dim = embedding_dim // num_heads
        self.rope = RoPE(embedding_dim, context_len)

        self.wq_d = nn.Linear(embedding_dim, q_latent_dim)
        self.w_qk = nn.Linear(q_latent_dim, num_heads * kv_latent_dim)

        self.wkv_d = nn.Linear(embedding_dim, kv_latent_dim)
        self.wv_u = nn.Linear(kv_latent_dim, num_heads * self.head_dim)

        self.w_out = nn.Linear(num_heads * self.head_dim, embedding_dim)

    def forward(self, x):
        b, seq_len, embedding_dim = x.shape
        c_q = self.rope(self.wq_d(x))
        c_kv = self.rope(self.wkv_d(x))

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


class FeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(embed_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        ff1_x = self.fc1(x)
        ff2_x= self.fc2(x)

        activation = F.silu(ff1_x) * ff2_x
        out = self.fc3(activation)
        return out


class MOELayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([FeedForwardLayer(embed_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(embed_dim, num_experts)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        hidden_state_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.router(hidden_state_reshaped)

        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(batch_size * seq_len, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        unique_experts = torch.unique(top_k_indices)
        for expert in unique_experts:
            expert_id = int(expert)

            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=1)
            assert token_mask.any()

            expert_input = hidden_state_reshaped[token_mask]
            expert_weights = top_k_probs[mask].unsqueeze(-1)
            expert_output = self.experts[expert_id](expert_input)

            output[token_mask] += expert_output * expert_weights

        output = output.view(batch_size, seq_len, hidden_dim)
        return output


class OutputLayer(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.output_layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim,
                 hidden_dim,
                 num_heads,
                 q_latent_dim,
                 kv_latent_dim,
                 num_experts = 256,
                 top_k = 8):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_dim)
        self.mla_layer = MultiHeadLatentAttention(embed_dim, num_heads, q_latent_dim, kv_latent_dim)

        self.sc = SkipConnection()
        self.rms_norm2 = RMSNorm(embed_dim)
        self.moe_layer = MOELayer(embed_dim, hidden_dim, num_experts, top_k)
        self.ffn = FeedForwardLayer(embed_dim, hidden_dim=18432)

    def forward(self, x, skip_moe = False):
        sc1 = x
        x = self.rms_norm1(x)
        x = self.mla_layer(x)
        x = self.sc(x, sc1)

        sc2 = x
        x = self.rms_norm2(x)

        if skip_moe:
            x = self.ffn(x)
        else:
            x = self.moe_layer(x)

        x = self.sc(x, sc2)
        return x


class DeepseekR1Model(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim,
                 hidden_dim,
                 num_heads,
                 q_latent_dim,
                 kv_latent_dim,
                 num_transformer_blocks = 61,
                 num_experts = 8):
        super().__init__()
        self.num_transformer_blocks = num_transformer_blocks
        self.input_layer = InputEmbedding(vocab_size, embed_dim)
        self.transformer_block = nn.Sequential(
            *[TransformerBlock(embed_dim, hidden_dim, num_heads, q_latent_dim, kv_latent_dim, num_experts) for _ in range(num_transformer_blocks)])

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x):
        x = self.input_layer(x)

        for i, block in enumerate(self.transformer_block):
            if i < 3:
                x = block(x, True)
            else:
                x = block(x)

        x = self.final_norm(x)
        x = self.output_layer(x)
        return x

