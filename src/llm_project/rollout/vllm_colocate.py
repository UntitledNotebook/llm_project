from __future__ import annotations

import time
from typing import Any

import torch

from llm_project.distributed import local_rank, world_size
from llm_project.rollout.formatting import encode_prompt_token_ids, format_rollout_batch
from llm_project.rollout.types import RolloutBackend, RolloutBatch, RolloutSamplingConfig
from llm_project.rollout.vllm_common import (
    flatten_vllm_outputs,
    generate_with_vllm,
    make_sampling_params,
    vllm_init_kwargs,
)
from llm_project.rollout.weight_sync import iter_model_weights, load_weights_into_vllm


class VLLMColocateRolloutBackend(RolloutBackend):
    name = "vllm_colocate"
    sync_method = "load_weights"

    def __init__(
        self,
        *,
        model_name_or_path: str,
        cfg: Any,
        tokenizer: Any,
        sampling: RolloutSamplingConfig,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
        from vllm import LLM

        self.tokenizer = tokenizer
        self.sampling = sampling
        self.importance_sampling = bool(sampling.request_logprobs)
        vllm_cfg = dict(cfg.rollout.get("vllm", {}))
        if world_size() > 1:
            vllm_cfg.setdefault("distributed_executor_backend", "external_launcher")
            vllm_cfg.setdefault("seed", int(cfg.seed))
        self.importance_sampling_cap = float(vllm_cfg.get("importance_sampling_cap", 3.0))
        self.llm = LLM(**vllm_init_kwargs(model_name_or_path, cfg, vllm_cfg))

    def sync_weights(self, model_engine: Any, *, step: int) -> float:
        start = time.perf_counter()
        load_weights_into_vllm(self.llm, iter_model_weights(model_engine, cpu=False))
        return time.perf_counter() - start

    def generate(self, prompts: list[str]) -> RolloutBatch:
        prompt_token_ids = encode_prompt_token_ids(
            self.tokenizer, prompts, self.sampling.max_prompt_length
        )
        sampling_params = make_sampling_params(self.sampling)
        outputs = generate_with_vllm(
            self.llm,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        completions = flatten_vllm_outputs(outputs)
        return format_rollout_batch(
            self.tokenizer,
            prompts,
            completions,
            max_prompt_length=self.sampling.max_prompt_length,
            group_size=self.sampling.group_size,
        )

