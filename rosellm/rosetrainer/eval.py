import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from config import GPTConfig
from dataset import FineWebNPYDataset, TextDatasetForCausalLM, build_tokenizer
from model import GPTModel
from torch.utils.data import DataLoader, DistributedSampler


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank: int) -> bool:
    return rank == 0


def setup_distributed_if_needed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"], 0)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False
    return device, rank, world_size, local_rank, distributed


def build_val_dataloader(
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
    world_size: int,
    rank: int,
):
    if args.data_mode == "text":
        if not args.val_data:
            raise ValueError("--val-data is not provided")
        tokenizer = build_tokenizer(args.tokenizer_name)
        val_dataset = TextDatasetForCausalLM(
            file_paths=args.val_data,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            add_eos=True,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    elif args.data_mode == "fineweb_npy":
        if not args.val_npy:
            raise ValueError("--val-npy is not provided")
        val_dataset = FineWebNPYDataset(
            file_paths=args.val_npy,
            seq_len=args.seq_len,
            max_tokens=args.max_tokens,
            seed=args.seed,
            random_start=True,
        )
    else:
        raise ValueError(f"invalid data mode: {args.data_mode}")

    if distributed:
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        sampler = None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is None else None,
    )
    return val_dataset, val_loader


def load_model_from_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
):
    ckpt = torch.load(args.checkpoint_path, map_location=device.type)
    cfg_dict = ckpt.get("config")
    if cfg_dict is not None:
        print("Found config in checkpoint, ignore cli configs")
        config = GPTConfig(**cfg_dict)
    else:
        print("No config found in checkpoint, use cli configs")
        config = GPTConfig()
    model = GPTModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    return model, config


@torch.no_grad()
def evaluate(
    model: GPTModel,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    distributed: bool,
    world_size: int,
):
    model.eval()
    total_loss_sum = torch.zeros(
        1,
        dtype=torch.float64,
        device=device,
    )
    total_tokens = torch.zeros(
        1,
        dtype=torch.float64,
        device=device,
    )
    from torch.amp import autocast

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if use_amp and device.type == "cuda":
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
        loss_sum = loss.detach().to(torch.float64) * batch_tokens
        total_loss_sum += loss_sum
        total_tokens += torch.tensor(
            float(batch_tokens),
            dtype=torch.float64,
            device=device,
        )
    if distributed:
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_sum.item() / max(total_tokens.item(), 1.0)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    args = parse_args()
    set_seed(args.seed)
    device, rank, world_size, local_rank, distributed = setup_distributed_if_needed()
    if is_main_process(rank):
        print(f"Eval device={device}")
        print(f"Eval world_size={world_size}")
        print(f"Eval ckpt: {args.checkpoint_path}")
    model, config = load_model_from_checkpoint(args, device)
    val_dataset, val_loader = build_val_dataloader(
        args, device, distributed, world_size, rank
    )
    use_amp = device.type == "cuda" and not args.no_amp
    if is_main_process(rank):
        print(f"use_amp: {use_amp}")
        print(f"val dataset size: {len(val_dataset)}")
    avg_loss, ppl = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        use_amp=use_amp,
        distributed=distributed,
        world_size=world_size,
    )
    if is_main_process(rank):
        print(f"avg_loss: {avg_loss}")
        print(f"ppl: {ppl}")
    if distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GPT model.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="text",
        choices=["text", "fineweb_npy"],
        help="data mode: text or fineweb_npy",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        nargs="*",
        default=[],
        help="Path to val data",
    )
    parser.add_argument(
        "--val-npy",
        type=str,
        nargs="*",
        default=[],
        help="path to val fineweb npy files",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="tokenizer name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="max tokens to sample from the dataset",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="disable amp",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
