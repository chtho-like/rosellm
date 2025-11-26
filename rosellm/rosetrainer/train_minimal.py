import torch
from config import GPTConfig
from model import GPTModel
from torch.utils.data import DataLoader, Dataset


class ToyRandomDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            dtype=torch.long,
        )
        labels = input_ids.clone()
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
    )
    model = GPTModel(config).to(device)
    dataset = ToyRandomDataset(
        vocab_size=config.vocab_size,
        seq_len=32,
        num_samples=1000,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    num_steps = 50
    step = 0
    for batch in dataloader:
        step += 1
        if step > num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step {step} / {num_steps} | loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
