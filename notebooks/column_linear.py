# import torch
# import torch.distributed as dist

# dist.init_process_group(backend="nccl")
# torch.cuda.set_device(dist.get_rank())
# rank = dist.get_rank()
# world_size = dist.get_world_size()
# if rank == 0:
#     X = torch.tensor([i for i in range(8)], dtype=torch.float).view(4, 2)
# else:
#     X = torch.zeros((4, 2), dtype=torch.float)

# print(f"step0: X on rank {rank}:\n{X}\n")

# X = X.to(torch.device("cuda", rank))

# dist.broadcast(X, src=0)

# print(f"step1: X broadcast to rank {rank}:\n{X}\n")

# W = torch.tensor([20 * rank + 10 * (i+1) for i in range(2)], dtype=torch.float)
# W = W.view(2, 1).to(torch.device("cuda", rank))

# print(f"step2: Weight W_{rank} on rank {rank}:\n{W}\n")

# Y_local = X @ W
# print(f"step3: Local result Y_{rank} on rank {rank}:\n{Y_local}\n")

# tensor_list = [torch.empty_like(Y_local) for _ in range(world_size)]
# dist.all_gather(tensor_list=tensor_list, tensor=Y_local)

# Y = torch.cat(tensor_list, dim=1)
# print(f"step4: Final output Y after all_gather on rank {rank}:\n{Y}\n")

# torchrun --nproc-per-node=2 column_linear.py
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
if rank == 0:
  X = torch.tensor([i for i in range(8)], dtype=torch.float).view(4, 2)
else:
  X = torch.zeros((4, 2), dtype=torch.float)
X = X.to(torch.device("cuda", rank))
dist.broadcast(X, src=0)
W = torch.tensor([20 * rank + 10 * (i+1) for i in range(2)], dtype=torch.float)
W = W.view(2, 1).to(torch.device("cuda", rank))
Y_ = X @ W
ls = [torch.empty_like(Y_) for _ in range(world_size)]
dist.all_gather(tensor_list=ls, tensor=Y_)
Y = torch.cat(ls, dim=1)
print(Y)