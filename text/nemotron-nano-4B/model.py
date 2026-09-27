import math

import torch
from torch import nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ffl1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffl2 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.ffl1(x)) ** 2
        return self.ffl2(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim

        self.w_q = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, num_groups * head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, num_groups * head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

    def forward(self, x, mask, kv_cache=None):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)

        
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.w_o(output), new_kv_cache


class Mamba2Block(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, num_groups, state_dim, conv_kernel):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel

        self.d_inner = num_heads * head_dim
        self.conv_dim = self.d_inner + 2 * num_groups * state_dim

        self.in_proj = nn.Linear(embed_dim, self.d_inner + self.conv_dim + num_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel,
                                groups=self.conv_dim, padding=conv_kernel - 1)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, num_heads + 1).float()))
        self.D = nn.Parameter(torch.ones(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, embed_dim, bias=False)

    def forward(self, x, ssm_cache=None):
        batch_size, seq_len, _ = x.shape

        z, xBC, dt = self.in_proj(x).split([self.d_inner, self.conv_dim, self.num_heads], dim=-1)

       
        if ssm_cache is not None:
            conv_state, h = ssm_cache
        else:
            conv_state = torch.zeros(batch_size, self.conv_kernel - 1, self.conv_dim, device=x.device)
            h = torch.zeros(batch_size, self.num_heads, self.head_dim, self.state_dim, device=x.device)

        xBC = torch.cat([conv_state, xBC], dim=1)
        new_conv_state = xBC[:, -(self.conv_kernel - 1):]
        xBC = self.conv1d(xBC.transpose(1, 2))[:, :, self.conv_kernel - 1:self.conv_kernel - 1 + seq_len].transpose(1, 2)
        xBC = F.silu(xBC)
        x_ssm, B, C = xBC.split([self.d_inner, self.num_groups * self.state_dim,
                                 self.num_groups * self.state_dim], dim=-1)

        x_ssm = x_ssm.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        B = B.reshape(batch_size, seq_len, self.num_groups, self.state_dim)
        C = C.reshape(batch_size, seq_len, self.num_groups, self.state_dim)
        reps = self.num_heads // self.num_groups
        B = B.repeat_interleave(reps, dim=2)
        C = C.repeat_interleave(reps, dim=2)

        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)

        outputs = []
        for t in range(seq_len):
            decay = torch.exp(dt[:, t] * A)
            update = dt[:, t, :, None, None] * x_ssm[:, t, :, :, None] * B[:, t, :, None, :]
            h = decay[:, :, None, None] * h + update
            outputs.append((h * C[:, t, :, None, :]).sum(dim=-1))
        y = torch.stack(outputs, dim=1) + self.D[None, None, :, None] * x_ssm
        y = y.reshape(batch_size, seq_len, self.d_inner)

        y = self.norm(y * F.silu(z))
        return self.out_proj(y), (new_conv_state, h)


class NemotronBlock(nn.Module):
    def __init__(self, mixer_type, has_ffn, embed_dim, num_heads, num_groups, head_dim,
                 mamba_num_heads, mamba_head_dim, mamba_num_groups, state_dim, conv_kernel, hidden_dim):
        super().__init__()
        self.mixer_type = mixer_type
        self.has_ffn = has_ffn

        self.rms_norm_1 = RMSNorm(embed_dim)
        if mixer_type == "M":
            self.mixer = Mamba2Block(embed_dim, mamba_num_heads, mamba_head_dim,
                                     mamba_num_groups, state_dim, conv_kernel)
        elif mixer_type == "*":
            self.mixer = GroupedQueryAttention(embed_dim, num_heads, num_groups, head_dim)
        else:
            raise ValueError(f"unknown mixer type: {mixer_type}")

        if has_ffn:
            self.rms_norm_2 = RMSNorm(embed_dim)
            self.ff = FeedForwardNetwork(embed_dim, hidden_dim)

    def forward(self, x, mask, cache=None):
        skip1 = x
        x = self.rms_norm_1(x)
        if self.mixer_type == "*":
            x, new_cache = self.mixer(x, mask, cache)
        else:
            x, new_cache = self.mixer(x, cache)
        x = x + skip1

        if self.has_ffn:
            skip2 = x
            x = self.rms_norm_2(x)
            x = self.ff(x)
            x = x + skip2

        return x, new_cache


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.output_layer(x)


class Cache:
    
    def __init__(self, num_blocks):
        self.layers = [None] * num_blocks
        self.num_tokens = 0 

class NemotronModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, pattern, num_heads, num_groups, head_dim,
                 mamba_num_heads, mamba_head_dim, mamba_num_groups,
                 state_dim, conv_kernel, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.input_embed = InputEmbedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList()
        for i, layer_type in enumerate(pattern):
            if layer_type == "-":
                continue
            has_ffn = i + 1 < len(pattern) and pattern[i + 1] == "-"
            self.blocks.append(
                NemotronBlock(layer_type, has_ffn, embed_dim, num_heads, num_groups, head_dim,
                              mamba_num_heads, mamba_head_dim, mamba_num_groups,
                              state_dim, conv_kernel, hidden_dim)
            )
        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, input_idx, cache=None):
        x = self.input_embed(input_idx)

       
        num_new = input_idx.shape[1]
        num_past = cache.num_tokens if cache is not None else 0
        mask = torch.tril(torch.ones(num_new, num_past + num_new, dtype=torch.bool, device=input_idx.device),
                          diagonal=num_past)

        for i, block in enumerate(self.blocks):
            block_cache = cache.layers[i] if cache is not None else None
            x, new_block_cache = block(x, mask, block_cache)
            if cache is not None:
                cache.layers[i] = new_block_cache

        if cache is not None:
            cache.num_tokens += num_new

        x = self.final_norm(x)
        x = self.output_layer(x)
        return x