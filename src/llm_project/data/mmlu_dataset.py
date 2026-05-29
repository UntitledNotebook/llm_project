from __future__ import annotations

from typing import Any

from datasets import load_dataset

# The common MMLU subject set used by cais/mmlu. You may reduce this list during debugging.
MMLU_SUBJECTS: list[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def resolve_subjects(subjects: str | list[str]) -> list[str]:
    if isinstance(subjects, str):
        if subjects.lower() == "all":
            return MMLU_SUBJECTS
        return [item.strip() for item in subjects.split(",") if item.strip()]
    return subjects


def load_mmlu_subject(
    dataset_name: str,
    subject: str,
    split: str = "test",
    max_samples: int | None = None,
):
    dataset = load_dataset(dataset_name, subject, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))
    return dataset

