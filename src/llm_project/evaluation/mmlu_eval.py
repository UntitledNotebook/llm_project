from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm_project.data.mmlu_dataset import load_mmlu_subject, mmlu_answer_index, resolve_subjects
from llm_project.data.prompts import build_mmlu_prompt


@torch.no_grad()
def _choice_loglikelihood(model: Any, tokenizer: Any, prompt: str, choice: str, device: torch.device) -> float:
    # Prefix a space so that answer choices are scored like natural continuations after "Answer:".
    choice_text = " " + choice.strip()
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    choice_ids = tokenizer(choice_text, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + choice_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = max(0, len(prompt_ids) - 1)
    return float(token_logps[0, start:].sum().item())


@torch.no_grad()
def evaluate_mmlu_model(
    model: Any,
    tokenizer: Any,
    *,
    dataset_name: str = "cais/mmlu",
    subjects: str | list[str] = "all",
    split: str = "test",
    max_samples_per_subject: int | None = None,
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

    for subject in tqdm(subject_list, desc="MMLU subjects"):
        dataset = load_mmlu_subject(dataset_name, subject, split=split, max_samples=max_samples_per_subject)
        subject_correct = 0
        for row in tqdm(dataset, desc=subject, leave=False):
            row = dict(row)
            choices = list(row["choices"])
            prompt = build_mmlu_prompt(subject, row["question"], choices)
            scores = [_choice_loglikelihood(model, tokenizer, prompt, letters[i], device) for i in range(len(choices))]
            pred_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
            gold_idx = mmlu_answer_index(row)
            correct = pred_idx == gold_idx
            subject_correct += int(correct)
            total_correct += int(correct)
            total_count += 1
            all_rows.append(
                {
                    "subject": subject,
                    "question": row["question"],
                    "prediction": letters[pred_idx],
                    "gold": letters[gold_idx],
                    "scores": scores,
                    "correct": correct,
                }
            )
        per_subject[subject] = {
            "total": len(dataset),
            "correct": subject_correct,
            "accuracy": subject_correct / max(1, len(dataset)),
        }

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
            json.dump({"metrics": metrics, "predictions": all_rows}, f, indent=2, ensure_ascii=False)
    return metrics
