import torch
import torch.distributed as dist


def init_process():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def example_all_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    gather_list = [
        torch.zeros(5, dtype=torch.float32).cuda() for _ in range(dist.get_world_size())
    ]
    print(f"Before all_gather on rank {dist.get_rank()}: {tensor}")
    dist.all_gather(gather_list, tensor)
    print(f"After all_gather on rank {dist.get_rank()}: {gather_list}")


init_process()
example_all_gather()
