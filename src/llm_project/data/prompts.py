from __future__ import annotations

from typing import Sequence


def build_math_sft_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Show the reasoning clearly and end with the final "
        "answer in the form `\\boxed{<answer>}`.\n\n"
        f"Problem:\n{problem.strip()}\n\nSolution:\n"
    )


def build_gsm8k_prompt(question: str) -> str:
    return (
        # "Solve the following grade-school math problem. Show your reasoning, then give the final "
        # "answer in the form `\\boxed{<answer>}`.\n\n"
        f"Problem:\n{question.strip()}\n\nSolution:\n"
    )


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
        "`\\boxed{C}` or `\\boxed{D}`.\n"
        "Show your reasoning, then end with exactly one boxed final answer.\n\n"
        f"Problem: {question.strip()}\n"
        f"{formatted_choices}\n\n"
        "Solution:\n"
    )
