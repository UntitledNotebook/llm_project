from __future__ import annotations

from dataclasses import dataclass

from llm_project.math_utils import verify_math_answer


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
        result = verify_math_answer(completion, reference_answer)
        correct = result.correct
        has_final_answer_marker = "####" in completion or "boxed" in completion.lower()
        format_ok = result.prediction is not None and has_final_answer_marker
        reward = self.correct_answer_reward if correct else self.wrong_answer_reward
        if self.format_reward and format_ok:
            reward += self.format_reward
        if self.require_final_answer_marker and not has_final_answer_marker:
            reward = self.wrong_answer_reward
        return RewardResult(
            reward=reward,
            correct=correct,
            pred_answer=result.prediction,
            reference_answer=result.reference,
            format_ok=format_ok,
        )
