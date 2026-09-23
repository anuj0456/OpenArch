import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import repeat_kv, KVCache


class GemmaRMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        std = x.pow(2).mean(dim=-1)
        variance = torch.sqrt(std + self.eps)
        x_norm = x / variance
        output = x_norm * self.weight
        return output.to(x.device)


class GemmaRoPE(nn.Module):
    def __init__(self, embed_dim, context_len, rope_theta=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = rope_theta if rope_theta > 0 else 10000
        inv_freq = (1. / N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(context_len)

        rotation_angle = position.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.register_buffer('cos', torch.cos(rotation_angle))
        self.register_buffer('sin', torch.sin(rotation_angle))

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        b, seq_len, embed_dim = x.shape

        adj_cos = self.cos[: seq_len,].unsqueeze(0).unsqueeze(0).to(x.dtype)
        adj_sin = self.sin[: seq_len,].unsqueeze(0).unsqueeze(0).to(x.dtype)

        rotation = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * adj_cos) + (adj_sin * rotation)
        return x_rotated


class GemmaMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x):
        up_proj = self.fc1(x)
        gate_proj = self.fc2(x)

        x = F.gelu(gate_proj) * up_proj
        down_proj = self.fc3(x)
        return down_proj


class GemmaAttention(nn.Module):
    def __init__(self, max_pos_emb, hidden_dim, num_heads, head_dim, num_kv_heads, rope_theta, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.max_pos_emb = max_pos_emb
        self.rope_theta = rope_theta

        self.num_kv_groups = num_heads // num_kv_heads
        self.is_causal = True

        assert self.hidden_dim % self.num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim)

        self.rope = GemmaRoPE(hidden_dim, max_pos_emb, rope_theta)

    def forward(self, hidden_states, mask=None, kv_cache:Optional[KVCache]=None):
        b, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_output = attn_score + mask
        attn_output = F.softmax(attn_output, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_output, v)


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(b, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output

class GemmaDecoder(nn.Module):
    def __init__(self, max_pos_emb, hidden_dim, num_heads, head_dim, num_kv_heads, rope_theta, layer_idx, intermediate_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn = GemmaAttention(max_pos_emb, hidden_dim, num_heads, head_dim, num_kv_heads, rope_theta, layer_idx)
        self.mlp = GemmaMLP(hidden_dim, intermediate_dim)
        self.input_norm = GemmaRMSNorm(hidden_dim)
        self.post_attn_norm = GemmaRMSNorm(hidden_dim)

    def forward(self, hidden_states, mask=None, position_id=None, kv_cache:Optional[KVCache]=None):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        hidden_states = self.attn(hidden_states, mask=mask, kv_cache=kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_state = residual + hidden_states
        return hidden_state


class GemmaModel(nn.Module):
    def __init__(self, vocab_size, max_pos_emb, hidden_dim, num_heads, head_dim, num_kv_heads, rope_theta, intermediate_dim, padding_idx, num_hidden_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.embed_dim = nn.Embedding(vocab_size, hidden_dim, self.padding_idx)
        self.layer = nn.ModuleList([
            GemmaDecoder(max_pos_emb, hidden_dim, num_heads, head_dim, num_kv_heads, rope_theta, layer_idx, intermediate_dim) for layer_idx in range(num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(hidden_dim)

    def get_input_embeddings(self):
        return self.embed_dim

    def forward(self, attention_mask, position_ids:Optional[KVCache]=None, input_embed=None, kv_cache=None):
        hidden_states = input_embed
        normalizer = torch.tensor(self.hidden_dim**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layer:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_ids, kv_cache=kv_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_embed, attention_mask, position_ids=None, kv_cache=None):
        output = self.model(input_embed, attention_mask, position_ids, kv_cache=kv_cache)

        hidden_states = output
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            'logits': logits,
        }

        if kv_cache is not None:
            return_data['kv_cache'] = kv_cache

        return return_data