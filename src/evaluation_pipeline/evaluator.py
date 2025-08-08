"""Evaluator orchestrating model inference, metrics and Opik logging."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm

from .config import EvaluationConfig
from .datasets import load_math_dataset, to_opik_items
from .metrics import compute_all_metrics
from .opik_client import OpikHelper


PredictFn = Callable[[str], str]


@dataclass
class EvalSample:
    dataset_item_id: str
    user_question: str
    expected_output: str


class Evaluator:
    """Run evaluation on a dataset and log to Opik."""

    def __init__(self, config: EvaluationConfig, predict_fn: PredictFn) -> None:
        self.config = config
        self.predict_fn = predict_fn
        self._opik = OpikHelper()

    def prepare_dataset(self) -> List[EvalSample]:
        # 1) Load HF dataset
        ds = load_math_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            subset=self.config.dataset_subset,
            max_samples=self.config.max_samples,
        )
        # 2) Upsert to Opik dataset and fetch ids
        items = to_opik_items(ds)
        dataset = self._opik.get_or_create_dataset(self.config.opik_dataset_name)
        self._opik.insert_items(dataset, items)
        # Re-fetch items (with ids)
        inserted = list(dataset.get_items())
        samples: List[EvalSample] = []
        for it in inserted:
            # Items may contain extra fields; we standardize
            q = it.get("user_question") or it.get("input")
            gt = it.get("expected_output") or it.get("reference")
            if not q or not gt:
                continue
            samples.append(EvalSample(dataset_item_id=it["id"], user_question=q, expected_output=gt))
        return samples

    def run(self) -> Dict[str, Any]:
        samples = self.prepare_dataset()
        bulk_items: List[Dict[str, Any]] = []

        for s in tqdm(samples, desc="Evaluating"):
            pred = self.predict_fn(s.user_question)
            metrics = compute_all_metrics(
                pred,
                s.expected_output,
                numeric_tol=self.config.numeric_tol,
                use_sympy=self.config.sympy_simplify,
            )
            # Prepare Opik payload
            feedback_scores = [
                {"name": k, "value": float(v.get("value", 0.0)), "source": "sdk"}
                for k, v in metrics.items()
            ]
            bulk_items.append(
                {
                    "dataset_item_id": s.dataset_item_id,
                    "evaluate_task_result": {
                        "prediction": pred,
                        "expected": s.expected_output,
                        "metrics": metrics,
                    },
                    "feedback_scores": feedback_scores,
                }
            )

        # Bulk log to Opik
        self._opik.bulk_log_experiment_items(
            experiment_name=self.config.experiment_name,
            dataset_name=self.config.opik_dataset_name,
            items=bulk_items,
        )

        # Aggregate metrics
        agg: Dict[str, float] = {}
        if bulk_items:
            # average by name
            sums: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for item in bulk_items:
                for score in item["feedback_scores"]:
                    name = score["name"]
                    val = float(score["value"])
                    sums[name] = sums.get(name, 0.0) + val
                    counts[name] = counts.get(name, 0) + 1
            for name, s in sums.items():
                agg[name] = s / max(1, counts[name])

        return {
            "num_samples": len(samples),
            "aggregated_metrics": agg,
            "experiment_name": self.config.experiment_name,
            "dataset_name": self.config.opik_dataset_name,
        }

