from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("VLLM_USE_V1", "0")

from llm_project.dtypes import vllm_dtype_from_str
from llm_project.rollout.types import RolloutSamplingConfig


def vllm_init_kwargs(model_name_or_path: str, cfg: Any, vllm_cfg: Any | None = None) -> dict[str, Any]:
    vllm_cfg = vllm_cfg or {}
    hf_home = os.environ.get("HF_HOME", "/mnt/zhangchenming/.cache/huggingface")
    download_dir = vllm_cfg.get("download_dir") or os.path.join(hf_home, "hub")
    kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "trust_remote_code": bool(cfg.model.get("trust_remote_code", True)),
        "dtype": vllm_dtype_from_str(cfg.model.get("dtype", "auto")),
        "tensor_parallel_size": int(vllm_cfg.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(vllm_cfg.get("gpu_memory_utilization", 0.8)),
        "download_dir": download_dir,
    }
    max_model_len = vllm_cfg.get("max_model_len")
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    max_num_seqs = vllm_cfg.get("max_num_seqs")
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    seed = vllm_cfg.get("seed", getattr(cfg, "seed", None))
    if seed is not None:
        kwargs["seed"] = int(seed)
    distributed_executor_backend = vllm_cfg.get("distributed_executor_backend")
    if distributed_executor_backend is not None:
        kwargs["distributed_executor_backend"] = str(distributed_executor_backend)
    return kwargs


def make_sampling_params(sampling: RolloutSamplingConfig):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "n": int(sampling.group_size),
        "max_tokens": int(sampling.max_new_tokens),
        "temperature": float(sampling.temperature if sampling.do_sample else 0.0),
        "top_p": float(sampling.top_p if sampling.do_sample else 1.0),
        "skip_special_tokens": True,
    }
    if sampling.request_logprobs:
        kwargs["logprobs"] = 0
    return SamplingParams(**kwargs)


def generate_with_vllm(
    llm: Any,
    *,
    prompt_token_ids: list[list[int]],
    sampling_params: Any,
) -> list[Any]:
    prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
    return llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)


def _sampled_token_logprobs(token_ids: list[int], logprob_steps: Any) -> list[float] | None:
    if logprob_steps is None:
        return None
    return [
        float(logprob_steps[token_idx][token_id].logprob)
        for token_idx, token_id in enumerate(token_ids)
    ]


def flatten_vllm_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    completions: list[dict[str, Any]] = []
    for request_output in outputs:
        for choice in request_output.outputs:
            token_ids = list(map(int, choice.token_ids))
            completions.append(
                {
                    "token_ids": token_ids,
                    "text": choice.text,
                    "logprobs": _sampled_token_logprobs(token_ids, choice.logprobs),
                }
            )
    return completions
