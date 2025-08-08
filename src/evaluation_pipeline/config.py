"""Evaluation configuration for math-focused LLMs with Opik."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class EvaluationConfig:
    """Config for evaluation runs."""

    # Dataset config
    dataset_name: str = "ngohongthai/exam-sixth_grade-instruct-dataset"
    dataset_split: str = "test"
    dataset_subset: Optional[str] = None
    max_samples: Optional[int] = None

    # Opik config
    opik_workspace: Optional[str] = field(default_factory=lambda: os.getenv("OPIK_WORKSPACE"))
    opik_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPIK_API_KEY"))
    opik_dataset_name: str = "mathpal-eval"
    experiment_name: str = "gemma3n-math-eval"
    project_name: Optional[str] = None

    # Task/model metadata (for experiment_config)
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    prompt_template: Optional[str] = None

    # Scoring
    metrics: List[str] = field(default_factory=lambda: [
        "exact_match",
        "normalized_numeric",
        "expression_equivalence",
        "format_validity",
    ])
    numeric_tol: float = 1e-6
    sympy_simplify: bool = True

    # Execution
    num_workers: int = 4
    batch_size: int = 8

    def to_experiment_config(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "revision": self.model_revision,
            "prompt_template": self.prompt_template,
            "dataset": self.dataset_name,
            "split": self.dataset_split,
            "subset": self.dataset_subset,
            "metrics": ",".join(self.metrics),
            "numeric_tol": self.numeric_tol,
        }

