import torch
from config import GPTConfig
from model import GPTModel


def main():
    config = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
    )
    model = GPTModel(config)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    logits = model(input_ids, attention_mask=attention_mask)
    print("input_ids shape:", input_ids.shape)  # [B, T]
    print("logits shape:", logits.shape)  # [B, T, V]


if __name__ == "__main__":
    main()
