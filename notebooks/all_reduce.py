import torch
import torch.distributed as dist


def init_process():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def example_all_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before all_reduce on rank {dist.get_rank()}: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce on rank {dist.get_rank()}: {tensor}")


init_process()
example_all_reduce()
