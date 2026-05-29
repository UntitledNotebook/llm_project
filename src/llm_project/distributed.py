from __future__ import annotations

from datetime import timedelta
import os
from typing import Any

import torch
import torch.distributed as dist


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return global_rank() == 0


def maybe_init_distributed() -> None:
    """Initialize torch.distributed when launched by deepspeed/torchrun."""
    if world_size() <= 1 or (dist.is_available() and dist.is_initialized()):
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA GPUs.")
    torch.cuda.set_device(local_rank())
    timeout_seconds = int(os.environ.get("LLM_PROJECT_DIST_TIMEOUT_SECONDS", "7200"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank()])
        else:
            dist.barrier()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def print_rank0(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs, flush=True)
