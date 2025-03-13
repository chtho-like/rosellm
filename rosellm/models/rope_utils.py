from typing import Optional, Tuple

import torch


def _compute_default_rope_parameters(
    rope_theta: float,
    head_dim: int,
    device: Optional[torch.device] = None,
    partial_rotary_factor: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    # E.g. 1000000.0 for Qwen2-0.5B
    base = rope_theta
    # E.g. 896 // 14 = 64 for Qwen2-0.5B
    dim = int(head_dim * partial_rotary_factor)
    # E.g. [0, 2, 4, 6, ..., 126] for Qwen2-0.5B
    i = torch.arange(0, dim, 2, dtype=torch.float32).to(device)
    # shape: [dim/2]
    # E.g. [32] for Qwen2-0.5B
    inv_freq = 1.0 / (base ** (i / dim))
    # actually not used in the default RoPE implementation
    attention_factor = 1.0
    return inv_freq, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}

if __name__ == "__main__":
    rope_theta = 10000
    head_dim = 64
    inv_freq, _ = _compute_default_rope_parameters(
        rope_theta=rope_theta,
        head_dim=head_dim,
    )
    print(inv_freq.shape)
    print(inv_freq)
