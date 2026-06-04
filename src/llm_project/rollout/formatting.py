from __future__ import annotations

from typing import Any

import torch

from llm_project.rollout.types import RolloutBatch


def completion_mask_from_eos(
    completion_ids: torch.Tensor, eos_token_id: int | None
) -> torch.Tensor:
    """Mask generated tokens through the first EOS, excluding right-padding after EOS."""
    mask = torch.ones_like(completion_ids, dtype=torch.long)
    if eos_token_id is None:
        return mask
    for row_idx in range(completion_ids.size(0)):
        eos_positions = (
            (completion_ids[row_idx] == int(eos_token_id))
            .nonzero(as_tuple=False)
            .flatten()
        )
        if eos_positions.numel() > 0:
            first_eos = int(eos_positions[0].item())
            if first_eos + 1 < completion_ids.size(1):
                mask[row_idx, first_eos + 1 :] = 0
    return mask


def encode_prompt_token_ids(
    tokenizer: Any, prompts: list[str], max_prompt_length: int
) -> list[list[int]]:
    old_truncation_side = getattr(tokenizer, "truncation_side", None)
    if old_truncation_side is not None:
        tokenizer.truncation_side = "left"
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=True,
        max_length=int(max_prompt_length),
        padding=False,
    )
    if old_truncation_side is not None:
        tokenizer.truncation_side = old_truncation_side
    return [list(map(int, ids)) for ids in encoded["input_ids"]]


def _completion_token_mask(completion_ids: list[int], eos_token_id: int | None) -> list[int]:
    if not completion_ids:
        return []
    ids = torch.tensor([completion_ids], dtype=torch.long)
    return completion_mask_from_eos(ids, eos_token_id).squeeze(0).tolist()


def format_rollout_batch(
    tokenizer: Any,
    prompts: list[str],
    completions: list[dict[str, Any]],
    *,
    max_prompt_length: int,
    group_size: int,
) -> RolloutBatch:

    group_size = int(group_size)
    expected = len(prompts) * group_size
    if len(completions) != expected:
        raise ValueError(f"Expected {expected} completions, got {len(completions)}")

    prompt_token_ids = encode_prompt_token_ids(tokenizer, prompts, max_prompt_length)
    repeated_prompt_ids = [ids for ids in prompt_token_ids for _ in range(group_size)]
    completion_token_ids = [list(map(int, item["token_ids"])) for item in completions]
    completion_texts = [item["text"] for item in completions]
    completion_logprobs = [item["logprobs"] for item in completions]

    pad_token_id = int(tokenizer.pad_token_id)
    eos_token_id = tokenizer.eos_token_id
    prompt_width = max(len(ids) for ids in repeated_prompt_ids)
    completion_width = max(len(ids) for ids in completion_token_ids)
    seq_len = prompt_width + completion_width

    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    completion_mask_rows: list[list[int]] = []
    sampling_rows: list[list[float]] | None = [] if completion_logprobs[0] is not None else None

    for prompt_ids, comp_ids, comp_logprobs in zip(
        repeated_prompt_ids, completion_token_ids, completion_logprobs
    ):
        prompt_pad = prompt_width - len(prompt_ids)
        completion_pad = completion_width - len(comp_ids)
        row = [pad_token_id] * prompt_pad + prompt_ids + comp_ids + [pad_token_id] * completion_pad
        attention = (
            [0] * prompt_pad
            + [1] * len(prompt_ids)
            + [1] * len(comp_ids)
            + [0] * completion_pad
        )
        comp_mask_values = _completion_token_mask(comp_ids, eos_token_id)
        comp_mask = [0] * prompt_width + comp_mask_values + [0] * completion_pad

        input_rows.append(row)
        attention_rows.append(attention)
        completion_mask_rows.append(comp_mask)

        if sampling_rows is not None:
            sampling = [0.0] * (seq_len - 1)
            for token_idx, value in enumerate(comp_logprobs):
                sampling[prompt_width + token_idx - 1] = float(value)
            sampling_rows.append(sampling)

    sampling_tensor = None
    if sampling_rows is not None:
        sampling_tensor = torch.tensor(sampling_rows, dtype=torch.float32)

    return RolloutBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        attention_mask=torch.tensor(attention_rows, dtype=torch.long),
        completion_mask=torch.tensor(completion_mask_rows, dtype=torch.long),
        completion_texts=completion_texts,
        sampling_logprobs=sampling_tensor,
        prompt_count=len(prompts),
        group_size=group_size,
    )
