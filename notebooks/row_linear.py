# import torch
# import torch.distributed as dist

# dist.init_process_group(backend="nccl")
# torch.cuda.set_device(dist.get_rank())
# rank = dist.get_rank()
# world_size = dist.get_world_size()
# if rank == 0:
#     XX = torch.tensor([i for i in range(8)], dtype=torch.float).view(4, 2).cuda(rank)
#     ls = [XX[:, i].clone().contiguous().view(4, 1) for i in range(2)]
# else:
#     ls = None

# X = torch.zeros((4, 1), dtype=torch.float, device=torch.device("cuda", rank))

# print(f"step0: ls on rank {rank}:\n{ls}\n")

# dist.scatter(X, scatter_list=ls, src=0)

# print(f"step1: X scatter to rank {rank}:\n{X}\n")

# W = torch.tensor([10 * (rank + 1) + 20 * i for i in range(2)], dtype=torch.float)
# W = W.view(1, 2).to(torch.device("cuda", rank))

# print(f"step2: Weight W_{rank} on rank {rank}:\n{W}\n")

# Y = X @ W
# print(f"step3: Local result Y_{rank} on rank {rank}:\n{Y}\n")

# dist.all_reduce(Y, op=dist.ReduceOp.SUM)

# print(f"step4: Final output Y after all_reduce on rank {rank}:\n{Y}\n")

# torchrun --nproc-per-node=2 row_linear.py
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
if rank == 0:
  XX = torch.tensor([i for i in range(8)], dtype=torch.float).view(4, 2).cuda(rank)
  ls = [XX[:, i].clone().contiguous().view(4, 1) for i in range(2)]
else:
  ls = None
X = torch.zeros((4, 1), dtype=torch.float, device=torch.device("cuda", rank))
dist.scatter(X, scatter_list=ls, src=0)
W = torch.tensor([10 * (rank+1) + 20*i for i in range(2)], dtype=torch.float)
W = W.view(1, 2).to(torch.device("cuda", rank))
Y = X @ W
dist.all_reduce(Y, op=dist.ReduceOp.SUM)
print(Y)