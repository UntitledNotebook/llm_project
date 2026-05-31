from __future__ import annotations

import os
from typing import Any


def _vllm_dtype(dtype: Any) -> str:
    if dtype is None:
        return "auto"
    normalized = str(dtype).lower()
    if normalized in {"auto", "none"}:
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16"
    if normalized in {"fp16", "float16", "half"}:
        return "float16"
    if normalized in {"fp32", "float32"}:
        return "float32"
    return str(dtype)


def load_vllm_llm(model_name_or_path: str, cfg: Any):
    from vllm import LLM

    vllm_cfg = cfg.get("vllm", {})
    download_dir = vllm_cfg.get("download_dir") or os.environ["HF_HOME"]
    kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "trust_remote_code": bool(cfg.model.get("trust_remote_code", True)),
        "dtype": _vllm_dtype(cfg.model.get("dtype", "auto")),
        "tensor_parallel_size": int(vllm_cfg.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
        "download_dir": download_dir,
    }
    max_model_len = vllm_cfg.get("max_model_len")
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    max_num_seqs = vllm_cfg.get("max_num_seqs")
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    return LLM(**kwargs)
