from __future__ import annotations

from collections.abc import Callable
from typing import Sequence


def _build_simple_gsm8k_prompt(problem: str) -> str:
    return f"Problem:\n{problem.strip()}\n\nSolution:\n"


def _build_step_by_step_gsm8k_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Please think step by step, show the reasoning "
        "clearly, and end with the final answer in the form `\\boxed{<answer>}`.\n\n"
        f"Problem:\n{problem.strip()}\n\nSolution:\n"
    )


_GSM8K_PROMPT_BUILDERS: dict[str, Callable[[str], str]] = {
    "simple": _build_simple_gsm8k_prompt,
    "step_by_step": _build_step_by_step_gsm8k_prompt,
}

def build_gsm8k_prompt(
    question: str,
    prompt_builder: str | None = "simple",
) -> str:
    builder_name = prompt_builder or "simple"
    return _GSM8K_PROMPT_BUILDERS[builder_name](question)


def normalize_subject_name(subject: str) -> str:
    return subject.replace("_", " ").strip()


def build_mmlu_prompt(subject: str, question: str, choices: Sequence[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    formatted_choices = "\n".join(f"{letters[i]}. {choice}" for i, choice in enumerate(choices))
    return (
        f"The following is a single-choice question about {normalize_subject_name(subject)}. "
        "There is exactly one correct answer.\n"
        "Choose from exactly four possible final answers: "
        "`\\boxed{A}`, `\\boxed{B}`, "
        "`\\boxed{C}`, `\\boxed{D}`.\n"
        "Show your reasoning, then end with exactly one boxed final answer.\n\n"
        f"Problem: {question.strip()}\n"
        f"{formatted_choices}\n\n"
        "Solution:\n"
    )
