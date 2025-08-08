"""Dataset preparation utilities for evaluation."""

from typing import Optional, Dict, Any, List
from datasets import load_dataset, Dataset


def load_math_dataset(
    dataset_name: str,
    split: str = "test",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Load HuggingFace dataset for evaluation.

    Expected fields: question, solution (ground-truth) or expected_output.
    """
    if subset:
        ds = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def to_opik_items(dataset: Dataset) -> List[Dict[str, Any]]:
    """Convert HF dataset rows to Opik dataset items schema.

    Schema used: {"user_question": str, "expected_output": str}
    """
    items: List[Dict[str, Any]] = []
    for row in dataset:
        question = row.get("question") or row.get("input")
        expected = row.get("solution") or row.get("expected_output")
        if question is None or expected is None:
            # skip invalid rows
            continue
        items.append({
            "user_question": str(question),
            "expected_output": str(expected),
        })
    return items

