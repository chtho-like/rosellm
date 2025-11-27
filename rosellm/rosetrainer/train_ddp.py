import argparse
import os

import torch
import torch.distributed as dist
from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from model import GPTModel
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(local_rank: int) -> bool:
    return local_rank == 0


def main(args: argparse.Namespace) -> None:
    device, local_rank = setup_distributed()
    checkpoint_path = args.checkpoint_path
    resume = args.resume
    if is_main_process(local_rank):
        print(f"[rank {local_rank}] Using device: {device}")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
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
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )
    dataset = ToyRandomDataset(
        vocab_size=config.vocab_size,
        seq_len=args.seq_len,
        num_samples=1000,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    ddp_model.train()
    num_steps = args.num_steps
    step = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"[rank {local_rank}] Resuming from checkpoint {checkpoint_path}")
        step, extra = load_checkpoint(
            checkpoint_path,
            ddp_model.module,
            optimizer,
            scaler,
            map_location=device.type,
        )
        print(f"[rank {local_rank}] Resumed from step {step}")
    elif resume and is_main_process(local_rank):
        print(
            f"[rank {local_rank}] Resume flag is set, but checkpoint not found. Starting from scratch."
        )
    elif is_main_process(local_rank):
        print(f"[rank {local_rank}] Starting from scratch")
    for epoch in range(1, 1000):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            step += 1
            if step > num_steps:
                break
            input_ids = batch["input_ids"].to(device)  # [B, T]
            labels = batch["labels"].to(device)  # [B, T]
            attention_mask = batch["attention_mask"].to(device)  # [B, T]
            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device.type):
                    logits, loss = ddp_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = ddp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss.backward()
                optimizer.step()
            if is_main_process(local_rank) and step % 20 == 0:
                save_checkpoint(
                    checkpoint_path,
                    model=ddp_model.module,
                    optimizer=optimizer,
                    step=step,
                    scaler=scaler if use_amp else None,
                    extra={"note": "minigpt_ddp"},
                )
            if is_main_process(local_rank) and step % 10 == 0:
                print(
                    f"[step {step} / {num_steps}] ",
                    f"loss = {loss.item():.4f} ",
                    f"amp = {use_amp}",
                )
        if step > num_steps:
            break
    if is_main_process(local_rank):
        print("Training finished.")
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training for GPT model.")
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
        help="Batch size per rank.",
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
        help="Total training steps.",
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
        default="checkpoints/minigpt_ddp.pt",
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
