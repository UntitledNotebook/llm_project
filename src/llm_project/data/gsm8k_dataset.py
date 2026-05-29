from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

from llm_project.data.prompts import build_gsm8k_prompt
from llm_project.math_utils import extract_answer


def load_gsm8k_raw(
    dataset_name: str = "openai/gsm8k",
    config_name: str = "main",
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, config_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))
    return dataset


class GSM8KPromptDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset) -> None:
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[int(index)]
        question = str(row["question"])
        answer_text = str(row["answer"])
        return {
            "question": question,
            "prompt": build_gsm8k_prompt(question),
            "answer": answer_text,
            "reference_answer": extract_answer(answer_text),
        }


def gsm8k_collate(features: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {key: [feature[key] for feature in features] for key in features[0]}
