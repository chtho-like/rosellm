from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    max_position_embeddings: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
