import os

import torch
from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from model import GPTModel
from torch.amp import GradScaler, autocast
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
    checkpoint_path = "checkpoints/minigpt_single.pt"
    resume = False
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
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    model.train()
    num_steps = 50
    step = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}")
        step, extra = load_checkpoint(checkpoint_path, model, optimizer, scaler)
        print(f"Resumed from step {step}")
    elif resume:
        print("Resume flag is set, but checkpoint not found. Starting from scratch.")
    else:
        print("Starting from scratch")
    for batch in dataloader:
        step += 1
        if step > num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast(device_type=device.type):
                logits, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss.backward()
            optimizer.step()
        if step % 20 == 0:
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                scaler=scaler if use_amp else None,
                extra={"note": "single_gpt_minimal"},
            )
        if step % 10 == 0:
            print(
                f"step {step} / {num_steps} ",
                f"loss: {loss.item():.4f} ",
                f"amp: {use_amp}",
            )


if __name__ == "__main__":
    main()
