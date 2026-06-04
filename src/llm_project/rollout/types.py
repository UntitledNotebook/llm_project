from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RolloutSamplingConfig:
    max_prompt_length: int
    max_new_tokens: int
    group_size: int
    temperature: float
    top_p: float
    do_sample: bool = True
    request_logprobs: bool = False


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    completion_texts: list[str]
    sampling_logprobs: torch.Tensor | None = None
    prompt_count: int = 0
    group_size: int = 1

    def to(self, device: torch.device | str) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            completion_mask=self.completion_mask.to(device),
            completion_texts=self.completion_texts,
            sampling_logprobs=(
                self.sampling_logprobs.to(device) if self.sampling_logprobs is not None else None
            ),
            prompt_count=self.prompt_count,
            group_size=self.group_size,
        )


class RolloutBackend(ABC):
    name: str
    sync_method: str
    importance_sampling: bool = False
    importance_sampling_cap: float = 1.0

    @abstractmethod
    def sync_weights(self, model_engine: Any, *, step: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompts: list[str]) -> RolloutBatch:
        raise NotImplementedError

    def close(self) -> None:
        return None
