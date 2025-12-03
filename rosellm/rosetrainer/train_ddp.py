import argparse
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from dataset import FineWebNPYDataset, TextDatasetForCausalLM, build_tokenizer
from model import GPTModel
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_line(path: str, text: str | tuple[str, ...]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {text}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(line) + "\n")
    print(line)


def evaluate_ddp(
    ddp_model: DDP,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model_was_training = ddp_model.module.training
    ddp_model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if use_amp:
                with autocast(device_type=device.type):
                    _, loss = ddp_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                _, loss = ddp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            batch_tokens = labels.numel()
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
    loss_tensor = torch.tensor(
        [total_loss, total_tokens],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    total_loss_all = float(loss_tensor[0].item())
    total_tokens_all = float(loss_tensor[1].item())
    avg_loss = total_loss_all / max(total_tokens_all, 1.0)
    if model_was_training:
        ddp_model.module.train()
    return avg_loss


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    warmup_steps: int,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    set_seed(args.seed)
    device, local_rank = setup_distributed()
    if args.use_wandb and is_main_process(local_rank):
        if wandb is None:
            raise ImportError("wandb is not installed")
        world_size = dist.get_world_size()
        wandb_config = {
            "world_size": world_size,
            "vocab_size": args.vocab_size,
            "max_position_embeddings": args.max_position_embeddings,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "use_tensor_parallel": args.use_tensor_parallel,
            "use_activation_checkpoint": args.use_activation_checkpoint,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_steps": args.num_steps,
            "lr": args.lr,
            "no_amp": args.no_amp,
            "checkpoint_path": args.checkpoint_path,
            "resume": args.resume,
            "lr_scheduler": args.lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "use_profiler": args.use_profiler,
            "train_data": args.train_data,
            "val_data": args.val_data,
            "tokenizer_name": args.tokenizer_name,
            "use_toy_data": args.use_toy_data,
            "max_tokens": args.max_tokens,
            "data_seed": args.data_seed,
            "eval_steps": args.eval_steps,
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )
    else:
        wandb_config = None
    if args.use_profiler:
        prof = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir"),
        )
        prof.__enter__()
        if is_main_process(local_rank):
            print("Profiler enabled. Trace will be saved after training")
    else:
        prof = None
    checkpoint_path = args.checkpoint_path
    resume = args.resume
    log_path = "logs/train_ddp.log"
    train_seed = args.data_seed
    val_seed = None if args.data_seed is None else args.data_seed + 1
    if is_main_process(local_rank):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(log_path, f"Training started at {timestamp}")
        log_line(log_path, f"[rank {local_rank}] Using device: {device}")
        log_line(log_path, f"Arguments: {args}")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if args.use_toy_data:
        effective_vocab_size = args.vocab_size
    else:
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
        use_tensor_parallel=args.use_tensor_parallel,
        use_activation_checkpoint=args.use_activation_checkpoint,
    )
    model = GPTModel(config).to(device)
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )
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
    elif args.data_mode == "text":
        train_dataset = TextDatasetForCausalLM(
            file_paths=args.train_data,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            add_eos=True,
            max_tokens=args.max_tokens,
            seed=train_seed,
        )
        if args.val_data:
            val_dataset = TextDatasetForCausalLM(
                file_paths=args.val_data,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                add_eos=True,
                max_tokens=args.val_max_tokens,
                seed=val_seed,
            )
        else:
            val_size = max(int(args.val_ratio * len(train_dataset)), 1)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
            )
    elif args.data_mode == "fineweb_npy":
        if not args.train_npy:
            raise ValueError("--train-npy is not provided")
        train_dataset = FineWebNPYDataset(
            file_paths=args.train_npy,
            seq_len=args.seq_len,
            max_tokens=args.max_tokens,
            seed=train_seed,
        )
        if args.val_npy:
            val_dataset = FineWebNPYDataset(
                file_paths=args.val_npy,
                seq_len=args.seq_len,
                max_tokens=args.val_max_tokens,
                seed=val_seed,
            )
        else:
            val_size = min(1024, max(int(args.val_ratio * len(train_dataset)), 1))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
            )
    else:
        raise ValueError(f"invalid data mode: {args.data_mode}")
    if is_main_process(local_rank):
        log_line(log_path, f"train dataset size: {len(train_dataset)}")
        log_line(log_path, f"val dataset size: {len(val_dataset)}")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    if is_main_process(local_rank):
        train_len = len(train_dataloader)
        steps_per_epoch = train_len // args.grad_accum_steps
        log_line(log_path, f"train micro steps per epoch: {train_len}")
        log_line(log_path, f"train steps per epoch: {steps_per_epoch}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    if args.lr_scheduler == "cosine":
        scheduler = build_lr_scheduler(
            optimizer,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
        )
    else:
        scheduler = None
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)
    ddp_model.train()
    num_steps = args.num_steps
    step = 0
    start_time = None
    last_log_time = None
    if resume and os.path.exists(checkpoint_path):
        log_line(
            log_path,
            f"[rank {local_rank}] Resuming from checkpoint {checkpoint_path}",
        )
        step, extra = load_checkpoint(
            checkpoint_path,
            ddp_model.module,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            map_location=device.type,
        )
        log_line(log_path, f"[rank {local_rank}] Resumed from step {step}")
        log_line(
            log_path,
            f"[rank {local_rank}] Reset optimizer lr to {args.lr}",
        )
    elif resume and is_main_process(local_rank):
        log_line(
            log_path,
            f"[rank {local_rank}] Resume flag is set, but checkpoint not found. Starting from scratch.",
        )
    elif is_main_process(local_rank):
        log_line(log_path, f"[rank {local_rank}] Starting from scratch")
    start_time = time.time()
    last_log_time = start_time
    start_step = step
    grad_accum_steps = max(1, args.grad_accum_steps)
    micro_step = 0
    for epoch in range(1, 1000):
        train_sampler.set_epoch(epoch)
        for batch in train_dataloader:
            if step >= num_steps:
                break
            micro_step += 1
            first_micro_step = (micro_step - 1) % grad_accum_steps == 0
            last_micro_step = micro_step % grad_accum_steps == 0
            if first_micro_step:
                t0 = time.time()
                optimizer.zero_grad()

            # data to device
            with record_function("data_to_device"):
                nvtx.range_push("data_to_device")
                input_ids = batch["input_ids"].to(device)  # [B, T]
                labels = batch["labels"].to(device)  # [B, T]
                attention_mask = batch["attention_mask"].to(device)  # [B, T]
                nvtx.range_pop()

            # forward
            with record_function("forward"):
                nvtx.range_push("forward")
                if use_amp:
                    with autocast(device_type=device.type):
                        logits, loss = ddp_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                else:
                    logits, loss = ddp_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                nvtx.range_pop()

            loss = loss / float(grad_accum_steps)

            # backward
            with record_function("backward"):
                nvtx.range_push("backward")
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                nvtx.range_pop()

            if last_micro_step:
                grad_norm = None
                with record_function("grad_clip_step"):
                    nvtx.range_push("grad_clip_step")
                    if args.grad_clip_norm > 0.0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(
                            ddp_model.parameters(),
                            args.grad_clip_norm,
                        )
                        grad_norm = float(grad_norm.item())
                    else:
                        total_norm = 0.0
                        for p in ddp_model.parameters():
                            if p.grad is None:
                                continue
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        grad_norm = total_norm**0.5
                    nvtx.range_pop()

                # optimizer step
                with record_function("optimizer_step"):
                    nvtx.range_push("optimizer_step")
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    nvtx.range_pop()

                if scheduler is not None:
                    scheduler.step()

                step += 1

                step_time = time.time() - t0
                tokens_per_step = input_ids.numel() * grad_accum_steps
                tokens_per_sec = tokens_per_step / max(step_time, 1e-8)

                if prof is not None:
                    prof.step()

                if is_main_process(local_rank) and step % 20 == 0:
                    save_checkpoint(
                        checkpoint_path,
                        model=ddp_model.module,
                        optimizer=optimizer,
                        step=step,
                        scaler=scaler if use_amp else None,
                        config=config,
                        scheduler=scheduler,
                        extra={"note": "minigpt_ddp"},
                    )
                if step % args.eval_steps == 0:
                    val_loss = evaluate_ddp(
                        ddp_model,
                        val_dataloader,
                        device=device,
                        use_amp=use_amp,
                    )
                    val_ppl = math.exp(val_loss)
                    if is_main_process(local_rank):
                        now = time.time()
                        steps_done = max(step - start_step, 1)
                        elapsed = now - start_time
                        time_since_last = now - last_log_time
                        avg_step_time = elapsed / steps_done
                        remaining_steps = max(num_steps - step, 0)
                        eta_seconds = remaining_steps * avg_step_time
                        last_log_time = now
                        current_lr = (
                            scheduler.get_last_lr()[0]
                            if scheduler is not None
                            else args.lr
                        )
                        msg = (
                            f"epoch {epoch} step {step} / {num_steps} ",
                            f"lr: {current_lr:.6f} ",
                            f"step time: {step_time:.2f}",
                            f"toks/s (per rank): {tokens_per_sec:.2f}",
                            f"grad norm: {grad_norm:.4f} ",
                            f"train loss: {loss.item():.4f} ",
                            f"val loss: {val_loss:.4f} ",
                            f"val ppl: {val_ppl:.4f} ",
                            f"dt: {time_since_last:.2f}s ",
                            f"eta: {eta_seconds/3600:.2f}h ",
                            f"amp: {use_amp}",
                        )
                        log_line(log_path, msg)
                        if args.use_wandb and wandb is not None:
                            world_size = dist.get_world_size()
                            global_toks_per_sec = tokens_per_sec * world_size
                            wandb.log(
                                {
                                    "train/loss": loss.item(),
                                    "val/loss": val_loss,
                                    "val/ppl": val_ppl,
                                    "grad_norm": grad_norm,
                                    "tokens_per_sec_per_rank": tokens_per_sec,
                                    "global_tokens_per_sec": global_toks_per_sec,
                                    "lr": current_lr,
                                    "amp": float(use_amp),
                                },
                                step=step,
                            )
        if step > num_steps:
            break
    if prof is not None:
        prof.__exit__(None, None, None)
    if is_main_process(local_rank):
        log_line(log_path, "Training finished.")
        if args.use_wandb and wandb is not None:
            wandb.finish()
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training for GPT model.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
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
        default=2,  # GPT2 small: 12
        help="Number of Transformer layers.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,  # GPT2 small: 12
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,  # GPT2 small: 768
        help="Model hidden size.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=512,  # GPT2 small: 3072
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
        "--use-activation-checkpoint",
        action="store_true",
        help="Use activation checkpointing.",
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
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps",
    )
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        help="Use profiler",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
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
        help="Path to val data",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,  # 10%
        help="Ratio of validation data",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluation step",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="text",
        choices=["text", "fineweb_npy"],
        help="data mode: text or fineweb_npy",
    )
    parser.add_argument(
        "--train-npy",
        type=str,
        nargs="*",
        default=[],
        help="path to training fineweb npy files",
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
        "--use-toy-data",
        action="store_true",
        help="use toy data",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="max tokens to sample from the dataset",
    )
    parser.add_argument(
        "--val-max-tokens",
        type=int,
        default=None,
        help="max tokens to sample from the val dataset",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="seed for the data sampler",
    )
    # wandb
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="use wandb for logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="rosetrainer",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="wandb run name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
