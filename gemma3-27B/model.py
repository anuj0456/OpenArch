import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, vocab = 262000, input_embed = 5376, ):
        super().__init__()
        self.input_embed = input_embed
        self.embeddings = nn.Embedding(vocab, input_embed)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.input_embed)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms

        out = x_norm * self.weight
        return out


class RoPE(nn.Module):
    def __init__(self, embed_dim, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        positions = torch.arange(context_len)

        rot_angle = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)

        self.cos = torch.cos(rot_angle)
        self.sin = torch.sin(rot_angle)

    def forward(self, x, offset):
        x1, x2 = x.chunk(2, dim=-1)

        adj_cos = self.cos[offset: offset+self.context_len, :].unsqueeze(0).unsqueeze(0)
        adj_sin = self.sin[offset: offset+self.context_len, :].unsqueeze(0).unsqueeze(0)

        rotation = torch.cat((-x2, x1), dim=-1)

        x_rotated = (adj_cos * x) + (adj_sin * rotation)

        return x_rotated

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, context_len, head_dim, num_kv_groups, num_heads = 32, qk_norm = False, query_pre_attn_scalar = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            head_dim = embed_dim // num_heads
        self.head_dim = head_dim

        self.out_embed = head_dim * num_heads

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        else:
            self.q_norm = self.k_norm = None

        self.rope = RoPE(embed_dim, context_len)

        self.w_q = nn.Linear(embed_dim, self.out_embed, bias = False)
        self.w_k = nn.Linear(embed_dim, head_dim * num_kv_groups, bias = False)
        self.w_v = nn.Linear(embed_dim, head_dim * num_kv_groups, bias = False)

        self.w_o = nn.Linear(self.out_embed, embed_dim, bias = False)

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar ** -0.5
        else:
            self.scaling = head_dim ** -0.5

    def forward(self, x, mask, start_pos=0, cache=None):
        batch, num_tokens, _ = x.shape

        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        query = query.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key_new = key.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        value_new = value.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key_new = self.k_norm(key_new)

        prev_len = 0
        if cache is not None:
            prev_k, prev_v = cache
            if prev_k is not None:
                prev_len = prev_k.size(2)
                keys_cat_raw = torch.cat([prev_k, key_new], dim = 2)
                values_cat_raw = torch.cat([prev_v, value_new], dim = 2)
            else:
                keys_cat_raw = key_new
                values_cat_raw = value_new
        else:
            keys_cat_raw = key_new
            values_cat_raw = value_new

        query = self.rope(query, start_pos)
        key = self.rope(keys_cat_raw, start_pos - prev_len)

        query = self.scaling * query

        if cache is not None and cache[0] is not None:
            next_cache = (
                torch.cat([cache[0], key_new], dim = 2),
                torch.cat([cache[1], value_new], dim = 2)
            )
        else:
            next_cache = (key_new, value_new)

        key = key.repeat_interleave(self.group_size, dim = 1)
        value = values_cat_raw.repeat_interleave(self.group_size, dim = 1)

        attn_score = query @ key.transpose(2, 3)
        attn_score = attn_score.masked_fill(mask, -torch.inf)
        attn_weights = F.softmax(attn_score, dim = -1)

        context = (attn_weights @ value).transpose(1, 2).reshape(batch, num_tokens, self.out_embed)

        output = self.w_o(context)
        return output, next_cache


class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim = 21504):
        super().__init__()
        self.ff1 = nn.Linear(embed_dim, hidden_dim)
        self.ff2 = nn.Linear(embed_dim, hidden_dim)
        self.ff3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        ff1_out = self.ff1(x)
        ff2_out = self.ff2(x)

        ff_activation = F.gelu(ff1_out)  * ff2_out

        ff_out = self.ff3(ff_activation)
        return ff_out


class OutputLayer(nn.Module):
    def __init__(self, out_dim = 5376, vocab_size = 262000):
        super().__init__()
        self.output_layer = nn.Linear(out_dim, vocab_size, bias = False)

    def forward(self, x):
        return self.output_layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, attn_type, input_embed, context_len, head_dim, num_kv_groups, num_heads, qk_norm, query_pre_attn_scalar, sliding_window):
        super().__init__()
        self.attn_type = attn_type
        self.sliding_window = sliding_window

        self.pre_norm1 = RMSNorm(input_embed)
        self.gpa_with_swa = GroupedQueryAttention(input_embed, context_len, head_dim, num_kv_groups, num_heads, qk_norm, query_pre_attn_scalar)
        self.post_norm1 = RMSNorm(input_embed)

        self.sc = SkipConnection()
        self.pre_norm2 = RMSNorm(input_embed)
        self.ff = FeedForward(input_embed)
        self.post_norm2 = RMSNorm(input_embed)

    def forward(self, x, mask_local, mask_global, start_pos, cache):
        sc1 = x
        x = self.pre_norm1(x)

        if self.attn_type == "sliding_window":
            if cache is not None and isinstance(cache, tuple):
                prev_k, _ = cache
                eff_kv_len = prev_k.size(2) + x.size(1)
            else:
                eff_kv_len = x.size(1)
            attn_mask = mask_local[..., -eff_kv_len:]
        else:
            attn_mask = mask_global

        x_attn, next_cache = self.gpa_with_swa.forward(x, attn_mask, start_pos, cache)
        if next_cache is not None and self.attn_type == "sliding_window":
            k,v = next_cache
            if k.size(2) > self.sliding_window:
                k = k[:, :, -self.sliding_window:, :]
                v = v[:, :, -self.sliding_window:, :]
            next_cache = (k, v)

        x_attn = self.post_norm1(x_attn)
        x = self.sc(sc1, x_attn)

        sc2 = x
        x = self.pre_norm2(x)
        x = self.ff(x)
        x = self.post_norm2(x)
        x = self.sc(sc2, x)

        return x, next_cache


class Gemma3Model(nn.Module):
    def __init__(self, vocab_size, input_embed, num_transformer_blocks, cfg):
        super().__init__()
        self.input_embed = InputEmbeddings(vocab_size, input_embed)
        self.transformer_blocks = nn.ModuleList(
            *[TransformerBlock(**cfg, attn_type = "sliding_window" if (i + 1) % 6 != 0 else "global") for i in range(num_transformer_blocks)],
        )
        self.final_norm = RMSNorm(input_embed)
        self.output_layer = OutputLayer(input_embed,  vocab_size)

        self.cfg = cfg
        self.current_pos = 0

    def _create_masks(self, cur_len, device, start_pos, end_pos):
        if end_pos is None:
            end_pos = cur_len
        total_len = end_pos

        ones = torch.ones((total_len, total_len), dtype=torch.bool, device=device)

        mask_global_full = torch.triu(ones, diagonal=1)
        far_past_full = torch.tril(ones, diagonal=self.cfg['sliding_window']).T
        mask_local_full = mask_global_full | far_past_full

        row_slice = slice(start_pos, end_pos)
        mask_global = mask_global_full[row_slice, :end_pos][None, None, :, :]
        mask_local = mask_local_full[row_slice, :end_pos][None, None, :, :]
        return mask_global, mask_local

    def forward(self, input_ids, cache):
        batch_size, seq_len = input_ids.shape
        x = self.input_embed(input_ids) * (self.cfg['embed_dim'] ** 0.5)

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + seq_len
            self.current_pos = pos_end

            mask_global, mask_local = self._create_masks(seq_len, input_ids.device, pos_start, pos_end)

        else:
            pos_start = 0
            mask_global, mask_local = self._create_masks(seq_len, input_ids.device, pos_start, seq_len)

        for i, block in enumerate(self.transformer_blocks):
            blck_cache = cache.get(i) if cache is not None else None
            x, new_blck_cache = block(x, mask_local, mask_global, pos_start, blck_cache)

            if cache is not None:
                cache.update(i, new_blck_cache)

        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits
