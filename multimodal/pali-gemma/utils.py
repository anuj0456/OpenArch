from typing import List
import torch


class KVCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def repeat_kv(hidden_state, n_rep: int):
    b, num_kv_heads, seq_len, head_dim = hidden_state.shape
    if n_rep == 1:
        return hidden_state
    hidden_state = hidden_state[:, :, None, :, :].expand(b, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_state.reshape(b, num_kv_heads * n_rep, seq_len, head_dim)