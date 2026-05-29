from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def strip_latex_delimiters(text: str) -> str:
    return text.replace("\\$", "").replace("\\%", "").replace("$", "").replace("\\,", "").replace("\\!", "").strip()


def extract_boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text or "")
    if matches:
        return strip_latex_delimiters(matches[-1])
    return None


def extract_after_final_marker(text: str) -> str | None:
    if not text:
        return None
    marker_patterns = [r"####\s*([^\n]+)", r"final answer is\s*[:\-]?\s*([^\n]+)", r"answer is\s*[:\-]?\s*([^\n]+)"]
    for pattern in marker_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return strip_latex_delimiters(matches[-1])
    return None


def extract_last_number(text: str) -> str | None:
    if not text:
        return None
    matches = _NUMBER_RE.findall(text.replace("−", "-"))
    return matches[-1] if matches else None


def extract_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    marked = extract_after_final_marker(text)
    if marked:
        last_in_marked = extract_last_number(marked)
        return last_in_marked or marked
    return extract_last_number(text)


def _to_decimal(value: str) -> Decimal | None:
    cleaned = strip_latex_delimiters(value).replace(",", "").replace("%", "").strip()
    cleaned = cleaned.rstrip(".")
    if not cleaned:
        return None
    try:
        if "/" in cleaned:
            return Decimal(Fraction(cleaned).numerator) / Decimal(Fraction(cleaned).denominator)
        return Decimal(cleaned)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    decimal = _to_decimal(value)
    if decimal is None:
        return strip_latex_delimiters(value).lower().strip()
    normalized = decimal.normalize()
    # Avoid scientific notation for simple integer answers.
    if normalized == normalized.to_integral():
        return format(normalized, "f").split(".", 1)[0]
    return format(normalized, "f").rstrip("0").rstrip(".")


def answers_match(prediction: str | None, reference: str | None) -> bool:
    pred_norm = normalize_numeric_answer(prediction)
    ref_norm = normalize_numeric_answer(reference)
    if pred_norm is None or ref_norm is None:
        return False
    if pred_norm == ref_norm:
        return True
    pred_dec = _to_decimal(pred_norm)
    ref_dec = _to_decimal(ref_norm)
    if pred_dec is not None and ref_dec is not None:
        return abs(pred_dec - ref_dec) <= Decimal("1e-4")
    return False
