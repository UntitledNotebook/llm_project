from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from llm_project.data.gsm8k_dataset import GSM8KPromptDataset, gsm8k_collate, load_gsm8k_raw
from llm_project.math_utils import verify_math_answer


@torch.no_grad()
def evaluate_gsm8k_model(
    model: Any,
    tokenizer: Any,
    *,
    dataset_name: str = "openai/gsm8k",
    config_name: str = "main",
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    raw = load_gsm8k_raw(dataset_name, config_name, split, max_samples)
    dataset = GSM8KPromptDataset(raw)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=gsm8k_collate
    )

    rows: list[dict[str, Any]] = []
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="GSM8K eval"):
        encoded = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        prompt_len = encoded.input_ids.size(1)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        completions = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for question, completion, answer_text in zip(
            batch["question"], completions, batch["answer"]
        ):
            result = verify_math_answer(completion, answer_text)
            is_correct = result.correct
            correct += int(is_correct)
            total += 1
            rows.append(
                {
                    "question": question,
                    "completion": completion,
                    "prediction": result.prediction,
                    "reference_answer": result.reference,
                    "reference_text": answer_text,
                    "correct": is_correct,
                }
            )
    tokenizer.padding_side = old_padding_side
    metrics = {"dataset": dataset_name, "split": split, "total": total, "correct": correct, "accuracy": correct / max(total, 1)}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": rows}, f, indent=2, ensure_ascii=False)
    return metrics
