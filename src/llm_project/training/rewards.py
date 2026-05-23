from __future__ import annotations

from dataclasses import dataclass

from llm_project.math_utils import answers_match, extract_answer


@dataclass
class RewardResult:
    reward: float
    correct: bool
    pred_answer: str | None
    reference_answer: str | None
    format_ok: bool


class GSM8KReward:
    def __init__(
        self,
        correct_answer_reward: float = 1.0,
        wrong_answer_reward: float = 0.0,
        format_reward: float = 0.0,
        require_final_answer_marker: bool = False,
    ) -> None:
        self.correct_answer_reward = float(correct_answer_reward)
        self.wrong_answer_reward = float(wrong_answer_reward)
        self.format_reward = float(format_reward)
        self.require_final_answer_marker = bool(require_final_answer_marker)

    def __call__(self, completion: str, reference_answer: str | None) -> RewardResult:
        pred_answer = extract_answer(completion)
        correct = answers_match(pred_answer, reference_answer)
        format_ok = "####" in completion or "boxed" in completion.lower() or "answer" in completion.lower()
        reward = self.correct_answer_reward if correct else self.wrong_answer_reward
        if self.format_reward and format_ok:
            reward += self.format_reward
        if self.require_final_answer_marker and "####" not in completion:
            reward = self.wrong_answer_reward
        return RewardResult(
            reward=reward,
            correct=correct,
            pred_answer=pred_answer,
            reference_answer=reference_answer,
            format_ok=format_ok,
        )
