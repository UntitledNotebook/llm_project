from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_from_str(dtype: str | None) -> torch.dtype | None:
    if dtype is None or str(dtype).lower() in {"auto", "none"}:
        return None
    normalized = str(dtype).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
    dtype: str | None = "bf16",
    attn_implementation: str | None = "flash_attention_2",
    device_map: str | dict[str, Any] | None = None,
    low_cpu_mem_usage: bool = True,
):
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype_from_str(dtype),
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if attn_implementation and attn_implementation.lower() not in {"none", "auto"}:
        kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def enable_training_mode(model) -> None:
    model.train()
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
