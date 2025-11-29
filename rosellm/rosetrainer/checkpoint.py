import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def _maybe_serialize_config(config: Any) -> Any:
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    return config


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    scaler: Optional["torch.amp.GradScaler"] = None,
    config: Any = None,
    scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if config is not None:
        ckpt["config"] = _maybe_serialize_config(config)
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
    scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
    map_location: Optional[str] = None,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    if map_location is None:
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    step = int(ckpt.get("step", 0))
    extra = ckpt.get("extra")
    return step, extra
