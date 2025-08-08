"""Math-focused metrics for LLM evaluation."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math
import re

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


@dataclass
class MetricResult:
    name: str
    value: float
    reason: Optional[str] = None


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_numbers(s: str) -> Optional[float]:
    """Extract the first floating-like number from text."""
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if match:
        try:
            return float(match.group(0))
        except Exception:
            return None
    return None


def metric_exact_match(prediction: str, reference: str) -> MetricResult:
    val = 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0
    return MetricResult(name="exact_match", value=val)


def metric_normalized_numeric(prediction: str, reference: str, tol: float = 1e-6) -> MetricResult:
    p_num = extract_numbers(prediction)
    r_num = extract_numbers(reference)
    if p_num is None or r_num is None:
        return MetricResult(name="normalized_numeric", value=0.0, reason="number_not_found")
    val = 1.0 if math.isclose(p_num, r_num, rel_tol=tol, abs_tol=tol) else 0.0
    return MetricResult(name="normalized_numeric", value=val)


def metric_expression_equivalence(prediction: str, reference: str, simplify: bool = True) -> MetricResult:
    if sp is None:
        return MetricResult(name="expression_equivalence", value=0.0, reason="sympy_not_available")
    try:
        p = sp.sympify(prediction)
        r = sp.sympify(reference)
        if simplify:
            val = 1.0 if sp.simplify(p - r) == 0 else 0.0
        else:
            val = 1.0 if sp.simplify(sp.Eq(p, r)) is True else 0.0
        return MetricResult(name="expression_equivalence", value=val)
    except Exception as e:
        return MetricResult(name="expression_equivalence", value=0.0, reason=str(e))


def metric_format_validity(prediction: str) -> MetricResult:
    """Check if prediction seems to be a concise math answer (heuristic)."""
    ok = bool(re.search(r"\d", prediction)) or bool(re.search(r"=", prediction))
    return MetricResult(name="format_validity", value=1.0 if ok else 0.0)


def compute_all_metrics(
    prediction: str,
    reference: str,
    *,
    numeric_tol: float = 1e-6,
    use_sympy: bool = True,
) -> Dict[str, Any]:
    results = {}
    for m in (
        metric_exact_match(prediction, reference),
        metric_normalized_numeric(prediction, reference, tol=numeric_tol),
        metric_expression_equivalence(prediction, reference, simplify=use_sympy),
        metric_format_validity(prediction),
    ):
        results[m.name] = {"value": m.value, **({"reason": m.reason} if m.reason else {})}
    return results

