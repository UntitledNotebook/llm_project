from __future__ import annotations

from typing import Any

import torch

_DTYPE_ALIASES: dict[str, tuple[torch.dtype, str]] = {
    "bf16": (torch.bfloat16, "bfloat16"),
    "bfloat16": (torch.bfloat16, "bfloat16"),
    "fp16": (torch.float16, "float16"),
    "float16": (torch.float16, "float16"),
    "half": (torch.float16, "float16"),
    "fp32": (torch.float32, "float32"),
    "float32": (torch.float32, "float32"),
    "float": (torch.float32, "float32"),
}


def _dtype_key(dtype: Any) -> str | None:
    if dtype is None:
        return None
    normalized = str(dtype).lower().removeprefix("torch.")
    if normalized in {"auto", "none"}:
        return None
    return normalized


def torch_dtype_from_str(dtype: Any) -> torch.dtype | None:
    key = _dtype_key(dtype)
    if key is None:
        return None
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _DTYPE_ALIASES[key][0]


def require_torch_dtype(dtype: Any) -> torch.dtype:
    torch_dtype = torch_dtype_from_str(dtype)
    if torch_dtype is None:
        raise ValueError(f"Expected a concrete dtype, got {dtype}")
    return torch_dtype


def vllm_dtype_from_str(dtype: Any) -> str:
    key = _dtype_key(dtype)
    if key is None:
        return "auto"
    if key not in _DTYPE_ALIASES:
        return str(dtype)
    return _DTYPE_ALIASES[key][1]
