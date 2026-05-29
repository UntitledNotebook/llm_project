from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

from llm_project.data.prompts import build_math_sft_prompt


@dataclass
class SFTExample:
    prompt: str
    response: str


def load_numina_gsm8k_sft_raw(
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    dataset = dataset.filter(lambda row: str(row.get("source", "")).lower() == "gsm8k")
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))
    return dataset


def train_validation_split(dataset: Dataset, validation_size: int, seed: int) -> tuple[Dataset, Dataset]:
    n_val = min(int(validation_size), max(1, len(dataset) // 10))
    split = dataset.train_test_split(test_size=n_val, seed=seed, shuffle=True)
    return split["train"], split["test"]


class SFTDataset(TorchDataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        tokenizer,
        *,
        max_seq_length: int,
    ) -> None:
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def _to_example(self, row: dict[str, Any]) -> SFTExample:
        problem = row.get("problem")
        response = row.get("solution")
        problem_text = str(problem).strip()
        response_text = str(response).strip()
        prompt = build_math_sft_prompt(problem_text)
        return SFTExample(prompt=prompt, response=response_text)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataset[int(index)]
        example = self._to_example(dict(row))
        prompt_ids = self.tokenizer(example.prompt, add_special_tokens=False).input_ids
        response_text = example.response
        if self.tokenizer.eos_token and not response_text.endswith(self.tokenizer.eos_token):
            response_text += self.tokenizer.eos_token
        response_ids = self.tokenizer(response_text, add_special_tokens=False).input_ids

        if len(response_ids) > self.max_seq_length:
            response_ids = response_ids[: self.max_seq_length]
        prompt_budget = self.max_seq_length - len(response_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTDataCollator:
    def __init__(self, tokenizer, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(feature["input_ids"].size(0) for feature in features)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            length = feature["input_ids"].size(0)
            pad_len = max_len - length
            batch["input_ids"].append(
                torch.cat([feature["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            batch["attention_mask"].append(
                torch.cat([feature["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat(
                    [
                        feature["labels"],
                        torch.full((pad_len,), self.label_pad_token_id, dtype=torch.long),
                    ]
                )
            )
        return {key: torch.stack(value, dim=0) for key, value in batch.items()}
