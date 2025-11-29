import argparse
import math
import os
from datetime import datetime

import torch
from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from dataset import TextDatasetForCausalLM, build_tokenizer
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


def log_line(path: str, text: str | tuple[str, ...]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(text) + "\n")
    print(text)


def evaluate(
    model: GPTModel,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if use_amp:
                with autocast(device_type=device.type):
                    _, loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                _, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            batch_tokens = labels.numel()
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    if model_was_training:
        model.train()
    return avg_loss


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = "logs/train_minimal.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line(log_path, f"Training started at {timestamp}")
    log_line(log_path, f"Using device: {device}")
    log_line(log_path, f"Arguments: {args}")
    checkpoint_path = args.checkpoint_path
    resume = args.resume

    if args.use_toy_data:
        effective_vocab_size = args.vocab_size
    else:
        if not args.train_data:
            raise ValueError("--train-data is not provided")
        tokenizer = build_tokenizer(args.tokenizer_name)
        tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
        if tokenizer_vocab_size is None:
            tokenizer_vocab_size = len(tokenizer)
        effective_vocab_size = tokenizer_vocab_size

    config = GPTConfig(
        vocab_size=effective_vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = GPTModel(config).to(device)

    if args.use_toy_data:
        full_dataset = ToyRandomDataset(
            vocab_size=config.vocab_size,
            seq_len=args.seq_len,
            num_samples=1000,
        )
        val_size = max(int(0.2 * len(full_dataset)), 1)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
        )
    else:
        train_dataset = TextDatasetForCausalLM(
            file_paths=args.train_data,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            add_eos=True,
            max_tokens=args.max_tokens,
            seed=args.data_seed,
        )
        if args.val_data:
            val_dataset = TextDatasetForCausalLM(
                file_paths=args.val_data,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                add_eos=True,
                max_tokens=args.max_tokens,
                seed=args.data_seed + 1,
            )
        else:
            val_size = max(int(0.1 * len(train_dataset)), 1)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
            )
    log_line(log_path, f"train dataset size: {len(train_dataset)}")
    log_line(log_path, f"val dataset size: {len(val_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)
    model.train()
    num_steps = args.num_steps
    step = 0
    if resume and os.path.exists(checkpoint_path):
        log_line(log_path, f"Resuming from checkpoint {checkpoint_path}")
        step, extra = load_checkpoint(checkpoint_path, model, optimizer, scaler)
        log_line(log_path, f"Resumed from step {step}")
    elif resume:
        log_line(
            log_path,
            "Resume flag is set, but checkpoint not found. Starting from scratch.",
        )
    else:
        log_line(log_path, "Starting from scratch")
    while step < num_steps:
        for batch in train_dataloader:
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
                val_loss = evaluate(
                    model,
                    val_dataloader,
                    device=device,
                    use_amp=use_amp,
                )
                val_ppl = math.exp(val_loss)
                msg = (
                    f"step {step} / {num_steps} ",
                    f"train loss: {loss.item():.4f} ",
                    f"val loss: {val_loss:.4f} ",
                    f"val ppl: {val_ppl:.4f} ",
                    f"amp: {use_amp}",
                )
                log_line(log_path, msg)


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
        default=10000,
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
    # data and tokenizer
    parser.add_argument(
        "--train-data",
        type=str,
        nargs="*",
        default=[],
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        nargs="*",
        default=[],
        help="Path to val data (optional, can be auto-split from train)",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="tokenizer name",
    )
    parser.add_argument(
        "--use-toy-data",
        action="store_true",
        help="use random toy data rather than real data",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="max tokens to sample from the dataset",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="seed for the data sampler",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
