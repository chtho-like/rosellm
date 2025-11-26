import os

import torch
import torch.distributed as dist
from config import GPTConfig
from model import GPTModel
from tensor_parallel import init_tensor_parallel


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def main():
    device, local_rank = setup_distributed()
    torch.manual_seed(123)  # for deterministic seed
    torch.cuda.manual_seed(123)  # for deterministic seed
    init_tensor_parallel()
    world_size = dist.get_world_size()
    if local_rank == 0:
        print("world_size:", world_size)
    config = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
        use_tensor_parallel=True,
    )
    model = GPTModel(config).to(device)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones(
        batch_size,
        seq_len,
        dtype=torch.long,
        device=device,
    )
    logits, loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    with torch.no_grad():
        logits_ref = logits.clone()
        dist.broadcast(logits_ref, src=0)
        diff = (logits_ref - logits).abs().max()
        diff_val = diff.item()
    print("max diff vs rank0:", diff_val)
    if local_rank == 0:
        print("input_ids shape:", input_ids.shape)
        print("logits shape:", logits.shape)
        print("loss:", loss.item())
    cleanup_distributed()


if __name__ == "__main__":
    main()
