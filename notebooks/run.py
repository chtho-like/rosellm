import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_blocking_p2p(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        dist.recv(tensor=tensor, src=0)
    print(f"Rank {rank} has data {tensor[0]}")


def run_nonblocking_p2p(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
        print(f"Rank {rank} sent data to rank 1")
    else:
        req = dist.irecv(tensor=tensor, src=0)
        print(f"Rank {rank} received data from rank 0")
    if req is not None:
        req.wait()
    print(f"Rank {rank} has data {tensor[0]}")

def run(rank, size):
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"Rank {rank} has data {tensor[0]}")

def init_process(rank, size, fn, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
