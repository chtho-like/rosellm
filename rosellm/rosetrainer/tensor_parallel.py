from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

_TP_GROUP: Optional[dist.ProcessGroup] = None


def init_tensor_parallel(tp_size: Optional[int] = None) -> None:
    global _TP_GROUP
    if not dist.is_initialized():
        raise RuntimeError("dist is not initialized")
    world_size = dist.get_world_size()
    if tp_size is None:
        tp_size = world_size
    if tp_size != world_size:
        raise NotImplementedError("currently we only support tp_size == world_size")
    _TP_GROUP = dist.group.WORLD


def get_tensor_parallel_group() -> dist.ProcessGroup:
    if _TP_GROUP is None:
        raise RuntimeError("tensor parallel group is not initialized")
    return _TP_GROUP


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("dist is not initialized")
        tp_group = get_tensor_parallel_group()
        tp_world_size = dist.get_world_size(tp_group)
        if out_features % tp_world_size != 0:
            raise ValueError("out_features must be divisible by tp_world_size")
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size
        self.out_per_rank = out_features // tp_world_size
        self.rank = dist.get_rank(tp_group)
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_rank))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y_local = y_local + self.bias
        if not self.gather_output or self.tp_world_size == 1:
            return y_local
        out_list = [torch.empty_like(y_local) for _ in range(self.tp_world_size)]
        dist.all_gather(out_list, y_local, group=self.tp_group)
        y = torch.cat(out_list, dim=-1)
        return y
