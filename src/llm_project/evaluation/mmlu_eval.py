from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from llm_project.data.prompts import build_mmlu_prompt
from llm_project.math_utils import extract_after_final_marker, extract_boxed_answer
from llm_project.data.mmlu_dataset import load_mmlu_subject, resolve_subjects

_MMLU_ANSWER_RE = re.compile(r"^([A-Da-d1-4])$")
_MMLU_NUMBER_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}


def normalize_mmlu_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    cleaned = answer.strip().strip("$`). \t\n\r")
    match = _MMLU_ANSWER_RE.fullmatch(cleaned)
    if not match:
        return None
    value = match.group(1).upper()
    return _MMLU_NUMBER_TO_LETTER.get(value, value)


def extract_mmlu_answer(completion: str) -> str | None:
    boxed = extract_boxed_answer(completion)
    if boxed is not None:
        return normalize_mmlu_answer(boxed)
    marked = extract_after_final_marker(completion)
    if marked is not None:
        return normalize_mmlu_answer(marked)
    return None


def mmlu_answers_match(completion: str, reference: str) -> bool:
    return extract_mmlu_answer(completion) == normalize_mmlu_answer(reference)


@torch.no_grad()
def evaluate_mmlu_model(
    model: Any,
    tokenizer: Any,
    *,
    dataset_name: str = "cais/mmlu",
    subjects: str | list[str] = "all",
    split: str = "test",
    max_samples_per_subject: int | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:

    model.eval()
    device = next(model.parameters()).device
    subject_list = resolve_subjects(subjects)
    all_rows: list[dict[str, Any]] = []
    per_subject: dict[str, dict[str, Any]] = {}
    total_correct = 0
    total_count = 0
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    batch_size = max(1, int(batch_size))

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for subject in tqdm(subject_list, desc="MMLU subjects"):
            dataset = load_mmlu_subject(
                dataset_name, subject, split=split, max_samples=max_samples_per_subject
            )
            subject_correct = 0
            subject_total = len(dataset)

            for start in tqdm(range(0, subject_total, batch_size), desc=subject, leave=False):
                batch_end = min(start + batch_size, subject_total)
                batch_rows = [dict(dataset[idx]) for idx in range(start, batch_end)]
                prompts: list[str] = []
                gold_letters: list[str] = []

                for row in batch_rows:
                    choices = list(row["choices"])
                    prompts.append(build_mmlu_prompt(subject, row["question"], choices))
                    gold_letters.append(letters[row["answer"]])

                encoded = tokenizer(
                    prompts,
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
                completions = tokenizer.batch_decode(
                    generated[:, prompt_len:], skip_special_tokens=True
                )

                for row, completion, gold in zip(batch_rows, completions, gold_letters):
                    prediction = extract_mmlu_answer(completion)
                    correct = prediction == gold
                    subject_correct += int(correct)
                    total_correct += int(correct)
                    total_count += 1
                    all_rows.append(
                        {
                            "subject": subject,
                            "question": row["question"],
                            "completion": completion,
                            "prediction": prediction,
                            "gold": gold,
                            "correct": correct,
                        }
                    )

            per_subject[subject] = {
                "total": subject_total,
                "correct": subject_correct,
                "accuracy": subject_correct / max(1, subject_total),
            }
    finally:
        tokenizer.padding_side = old_padding_side

    metrics = {
        "dataset": dataset_name,
        "split": split,
        "total": total_count,
        "correct": total_correct,
        "accuracy": total_correct / max(total_count, 1),
        "per_subject": per_subject,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            payload = {"metrics": metrics, "predictions": all_rows}
            json.dump(payload, f, indent=2, ensure_ascii=False)
    return metrics
