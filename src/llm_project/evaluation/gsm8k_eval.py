from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm_project.data.gsm8k_dataset import GSM8KPromptDataset, load_gsm8k_raw
from llm_project.math_utils import verify_math_answer


def evaluate_gsm8k_model(
    llm: Any,
    *,
    dataset_name: str = "openai/gsm8k",
    config_name: str = "main",
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    prompt_builder: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    from vllm import SamplingParams

    raw = load_gsm8k_raw(dataset_name, config_name, split, max_samples)
    dataset = GSM8KPromptDataset(raw, prompt_builder=prompt_builder)
    batch_size = max(1, int(batch_size))
    sampling_params = SamplingParams(
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        skip_special_tokens=True,
    )

    rows: list[dict[str, Any]] = []
    correct = 0
    total = 0
    for start in tqdm(range(0, len(dataset), batch_size), desc="GSM8K eval"):
        batch_rows = [
            dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))
        ]
        prompts = [row["prompt"] for row in batch_rows]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        completions = [output.outputs[0].text if output.outputs else "" for output in outputs]
        for row, completion in zip(batch_rows, completions):
            answer_text = row["answer"]
            result = verify_math_answer(completion, answer_text)
            is_correct = result.correct
            correct += int(is_correct)
            total += 1
            rows.append(
                {
                    "question": row["question"],
                    "completion": completion,
                    "prediction": result.prediction,
                    "reference_answer": result.reference,
                    "reference_text": answer_text,
                    "correct": is_correct,
                }
            )
    metrics = {
        "dataset": dataset_name,
        "split": split,
        "prompt_builder": dataset.prompt_builder,
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": rows}, f, indent=2, ensure_ascii=False)
    return metrics
