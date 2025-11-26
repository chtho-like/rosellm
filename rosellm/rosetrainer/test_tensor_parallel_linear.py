import os

import torch
import torch.distributed as dist
import torch.nn as nn
from tensor_parallel import ColumnParallelLinear, init_tensor_parallel


def setup_distributed() -> torch.device:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def main() -> None:
    device = setup_distributed()
    init_tensor_parallel()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank == 0:
        print(f"world_size = {world_size}")
    batch_size = 4
    in_features = 8
    out_features = 12
    if out_features % world_size != 0:
        raise RuntimeError("out_features must be divisible by world_size")
    torch.manual_seed(42)
    ref_linear = nn.Linear(in_features, out_features, bias=True).to(device)
    tp_linear = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        gather_output=True,
    ).to(device)
    with torch.no_grad():
        out_per_rank = out_features // world_size
        start = rank * out_per_rank
        end = start + out_per_rank
        tp_linear.weight.copy_(ref_linear.weight[start:end, :])
        tp_linear.bias.copy_(ref_linear.bias[start:end])
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, device=device)
    y_ref = ref_linear(x)
    y_tp = tp_linear(x)
    diff = (y_ref - y_tp).abs().max()
    diff_val = diff.item()
    if rank == 0:
        print("max |y_ref - t_tp| = ", diff_val)
    cleanup_distributed()


if __name__ == "__main__":
    main()
