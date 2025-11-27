import os

import torch
import torch.distributed as dist
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


def main():
    device, local_rank = setup_distributed()
    if is_main_process(local_rank):
        print(f"[rank {local_rank}] Using device: {device}")
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
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )
    dataset = ToyRandomDataset(
        vocab_size=config.vocab_size,
        seq_len=32,
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
        batch_size=8,
        sampler=sampler,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    ddp_model.train()
    num_steps = 50
    step = 0
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


if __name__ == "__main__":
    main()
