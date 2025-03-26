import torch
import torch.distributed as dist
def init_process():
  dist.init_process_group(backend='nccl')
  torch.cuda.set_device(dist.get_rank())
def example_gather():
  tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
  if dist.get_rank() == 0:
    gather_list = [
      torch.zeros(5, dtype=torch.float32).cuda()
      for _ in range(dist.get_world_size())
    ]
  else:
    gather_list = None
  print(f"Before gather on rank {dist.get_rank()}: {tensor}")
  dist.gather(tensor, gather_list, dst=0)
  if dist.get_rank() == 0:
    print(f"After gather on rank 0: {gather_list}")
init_process()
example_gather()
