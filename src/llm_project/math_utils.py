from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

_GSM8K_FINAL_MARKER = "####"

_PREDICTION_EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        boxed_match_priority=0,
        normalization_config=NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=False,
            nits=False,
            boxed="all",
            equations=False,
        ),
    ),
    ExprExtractionConfig(),
]
_REFERENCE_EXTRACTION_CONFIG = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]


@dataclass(frozen=True)
class MathVerificationResult:
    correct: bool
    prediction: str | None
    reference: str | None
    prediction_parsed: list[Any]
    reference_parsed: list[Any]


def _gsm8k_reference_text(text: str) -> str:
    if _GSM8K_FINAL_MARKER not in text:
        return text
    return text.rsplit(_GSM8K_FINAL_MARKER, maxsplit=1)[-1].strip()


def parse_math_answer(text: str | None, *, is_prediction: bool) -> list[Any]:
    if not text:
        return []
    text_to_parse = text if is_prediction else _gsm8k_reference_text(text)
    return parse(
        text_to_parse,
        extraction_config=(
            _PREDICTION_EXTRACTION_CONFIG if is_prediction else _REFERENCE_EXTRACTION_CONFIG
        ),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
    )


def parsed_answer_to_string(parsed: Sequence[Any]) -> str | None:
    if not parsed:
        return None
    if len(parsed) == 1:
        return str(parsed[0])
    return ", ".join(str(item) for item in parsed)


def verify_math_answer(completion: str, reference: str | None) -> MathVerificationResult:
    prediction_parsed = parse_math_answer(completion, is_prediction=True)
    reference_parsed = parse_math_answer(reference, is_prediction=False)
    correct = bool(prediction_parsed) and bool(reference_parsed) and verify(
        reference_parsed,
        prediction_parsed,
    )
    return MathVerificationResult(
        correct=correct,
        prediction=parsed_answer_to_string(prediction_parsed),
        reference=parsed_answer_to_string(reference_parsed),
        prediction_parsed=prediction_parsed,
        reference_parsed=reference_parsed,
    )
