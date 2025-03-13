from typing import List, Tuple

import torch


class Cache(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class DynamicCache(Cache):
    def __init__(self):
        super().__init__()
        self._seen_tokens = 0
        # keyed by layer index
        # [[batch_size, num_key_value_heads, seq_len, head_dim], ...]
        self.key_cache: List[torch.Tensor] = []
        # [[batch_size, num_key_value_heads, seq_len, head_dim], ...]
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        # [batch_size, num_key_value_heads, seq_len, head_dim]
        key_states: torch.Tensor,
        # [batch_size, num_key_value_heads, seq_len, head_dim]
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.empty_like(key_states))
                self.value_cache.append(torch.empty_like(value_states))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [
                    self.key_cache[layer_idx],
                    key_states,
                ],
                dim=-2,
            )
            self.value_cache[layer_idx] = torch.cat(
                [
                    self.value_cache[layer_idx],
                    value_states,
                ],
                dim=-2,
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
