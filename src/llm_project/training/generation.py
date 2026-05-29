from __future__ import annotations

from typing import Any

import torch


def completion_mask_from_eos(completion_ids: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    """Mask generated tokens through the first EOS, excluding right-padding after EOS."""
    mask = torch.ones_like(completion_ids, dtype=torch.long)
    if eos_token_id is None:
        return mask
    for row_idx in range(completion_ids.size(0)):
        eos_positions = (completion_ids[row_idx] == int(eos_token_id)).nonzero(as_tuple=False).flatten()
        if eos_positions.numel() > 0:
            first_eos = int(eos_positions[0].item())
            if first_eos + 1 < completion_ids.size(1):
                mask[row_idx, first_eos + 1 :] = 0
    return mask


@torch.no_grad()
def generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    group_size: int,
    temperature: float,
    top_p: float,
    do_sample: bool = True,
) -> dict[str, Any]:
    """Generate grouped completions using the local HF model copy."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = encoded["input_ids"].size(1)
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    completion_ids = generated[:, prompt_len:]
    completion_mask = completion_mask_from_eos(completion_ids, tokenizer.eos_token_id)
    prompt_attention_mask = encoded["attention_mask"].repeat_interleave(group_size, dim=0)
    completion_attention_mask = torch.ones_like(completion_mask, dtype=torch.long)
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    full_completion_mask = torch.cat(
        [torch.zeros_like(prompt_attention_mask, dtype=torch.long), completion_mask], dim=1
    )
    decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    tokenizer.padding_side = old_padding_side
    if was_training:
        model.train()
    return {
        "input_ids": generated,
        "attention_mask": attention_mask,
        "completion_mask": full_completion_mask,
        "completion_texts": decoded,
    }
