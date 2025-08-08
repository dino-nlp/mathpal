"""Opik client helpers for datasets and experiment logging."""

from typing import List, Dict, Any, Optional


class OpikHelper:
    """Lightweight wrapper around Opik Python SDK for common ops."""

    def __init__(self) -> None:
        try:
            import opik  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("opik is required. Install with `pip install opik`. ") from e
        self._opik = opik
        self._client = opik.Opik()

    @property
    def client(self):
        return self._client

    def get_or_create_dataset(self, name: str):
        return self._client.get_or_create_dataset(name)

    def insert_items(self, dataset, items: List[Dict[str, Any]]) -> None:
        # Opik de-duplicates automatically
        dataset.insert(items)

    def bulk_log_experiment_items(
        self,
        experiment_name: str,
        dataset_name: str,
        items: List[Dict[str, Any]],
    ) -> None:
        # Items schema: { dataset_item_id, evaluate_task_result, feedback_scores }
        self._client.rest_client.experiments.experiment_items_bulk(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            items=items,
        )

