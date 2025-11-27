import argparse
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


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    checkpoint_path = args.checkpoint_path
    resume = args.resume
    config = GPTConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = GPTModel(config).to(device)
    dataset = ToyRandomDataset(
        vocab_size=config.vocab_size,
        seq_len=args.seq_len,
        num_samples=1000,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)
    model.train()
    num_steps = args.num_steps
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train minimal GPT model.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=128,
        help="Max sequence length.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of Transformer layers.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model hidden size.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=512,
        help="FFN hidden size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--use-tensor-parallel",
        action="store_true",
        help="Enable tensor parallel blocks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per step.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP even on CUDA.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/minigpt_single.pt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
