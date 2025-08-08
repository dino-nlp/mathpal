"""Evaluation pipeline for math-focused LLMs using Opik.

This package provides:
- Config management for evaluation runs
- Dataset preparation utilities (HF datasets or custom)
- Math-focused metrics (exact match, numeric match, expression equivalence)
- Opik integration for datasets, experiments and bulk logging
- Orchestrated evaluator and a CLI entrypoint
"""

from .config import EvaluationConfig
from .evaluator import Evaluator

__all__ = [
    "EvaluationConfig",
    "Evaluator",
]

