from __future__ import annotations

from typing import Any

import torch

from llm_project.rollout.formatting import completion_mask_from_eos
from llm_project.rollout.types import RolloutBackend, RolloutBatch, RolloutSamplingConfig


@torch.no_grad()
def _generate_hf_rollout(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    sampling: RolloutSamplingConfig,
) -> RolloutBatch:
    was_training = model.training
    old_padding_side = getattr(tokenizer, "padding_side", None)
    old_truncation_side = getattr(tokenizer, "truncation_side", None)
    model.eval()
    try:
        if old_padding_side is not None:
            tokenizer.padding_side = "left"
        if old_truncation_side is not None:
            tokenizer.truncation_side = "left"
        device = next(model.parameters()).device
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=sampling.max_prompt_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_len = encoded["input_ids"].size(1)
        generate_kwargs = {
            **encoded,
            "max_new_tokens": sampling.max_new_tokens,
            "do_sample": sampling.do_sample,
            "num_return_sequences": sampling.group_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if sampling.do_sample:
            generate_kwargs["temperature"] = sampling.temperature
            generate_kwargs["top_p"] = sampling.top_p
        generated = model.generate(**generate_kwargs)
        completion_ids = generated[:, prompt_len:]
        completion_mask = completion_mask_from_eos(completion_ids, tokenizer.eos_token_id)
        prompt_attention_mask = encoded["attention_mask"].repeat_interleave(
            sampling.group_size, dim=0
        )
        completion_attention_mask = completion_mask.to(dtype=torch.long)
        attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
        full_completion_mask = torch.cat(
            [torch.zeros_like(prompt_attention_mask, dtype=torch.long), completion_mask], dim=1
        )
        return RolloutBatch(
            input_ids=generated,
            attention_mask=attention_mask,
            completion_mask=full_completion_mask,
            completion_texts=tokenizer.batch_decode(completion_ids, skip_special_tokens=True),
            sampling_logprobs=None,
            prompt_count=len(prompts),
            group_size=sampling.group_size,
        )
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side
        if old_truncation_side is not None:
            tokenizer.truncation_side = old_truncation_side
        if was_training:
            model.train()


class HFRolloutBackend(RolloutBackend):
    name = "hf"
    sync_method = "none"
    importance_sampling = False
    importance_sampling_cap = 1.0

    def __init__(
        self, *, model_engine: Any, tokenizer: Any, sampling: RolloutSamplingConfig
    ) -> None:
        self.model_engine = model_engine
        self.tokenizer = tokenizer
        self.sampling = sampling

    def sync_weights(self, model_engine: Any, *, step: int) -> float:
        self.model_engine = model_engine
        return 0.0

    def generate(self, prompts: list[str]) -> RolloutBatch:
        return _generate_hf_rollout(
            self.model_engine.module,
            self.tokenizer,
            prompts,
            self.sampling,
        )

