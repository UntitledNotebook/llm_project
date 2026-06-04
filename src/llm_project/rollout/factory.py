from __future__ import annotations

from typing import Any

import torch

from llm_project.rollout.hf_backend import HFRolloutBackend
from llm_project.rollout.types import RolloutBackend, RolloutSamplingConfig
from llm_project.rollout.vllm_colocate import VLLMColocateRolloutBackend
from llm_project.rollout.vllm_server_client import VLLMServerRolloutBackend


def rollout_sampling_from_config(cfg: Any) -> RolloutSamplingConfig:
    vllm_cfg = cfg.rollout.get("vllm", {})
    backend = str(cfg.rollout.get("backend", "hf"))
    request_logprobs = backend.startswith("vllm") and bool(
        vllm_cfg.get("importance_sampling", True)
    )
    return RolloutSamplingConfig(
        max_prompt_length=int(cfg.dataset.max_prompt_length),
        max_new_tokens=int(cfg.rollout.max_new_tokens),
        group_size=int(cfg.rollout.group_size),
        temperature=float(cfg.rollout.temperature),
        top_p=float(cfg.rollout.top_p),
        do_sample=bool(cfg.rollout.do_sample),
        request_logprobs=request_logprobs,
    )


def create_rollout_backend(
    *,
    cfg: Any,
    model_name_or_path: str,
    tokenizer: Any,
    model_engine: Any,
    device: torch.device,
) -> RolloutBackend:
    backend = str(cfg.rollout.get("backend", "hf"))
    sampling = rollout_sampling_from_config(cfg)
    if backend == "hf":
        return HFRolloutBackend(model_engine=model_engine, tokenizer=tokenizer, sampling=sampling)
    if backend == "vllm_colocate":
        return VLLMColocateRolloutBackend(
            model_name_or_path=model_name_or_path,
            cfg=cfg,
            tokenizer=tokenizer,
            sampling=sampling,
        )
    if backend == "vllm_server":
        server_backend = VLLMServerRolloutBackend(
            cfg=cfg, tokenizer=tokenizer, sampling=sampling, device=device
        )
        server_backend.health()
        return server_backend
    raise ValueError(f"Unsupported rollout.backend: {backend}")
