import argparse

from config import settings
from core.logger_utils import get_logger
from core.opik_utils import create_dataset_from_artifacts
from mathpal import MathPal
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, LevenshteinRatio, Moderation

from .style import Style

logger = get_logger(__name__)


def make_evaluation_task(inference_pipeline: MathPal):
    def evaluation_task(x: dict) -> dict:
        answer = inference_pipeline.generate(
            question=x["question"],
            sample_for_evaluation=True,
        )["answer"]

        return {
            "input": x["question"],
            "output": answer,
            "expected_output": x["solution"],
            "reference": x["solution"],
        }

    return evaluation_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate monitoring script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mathpal-testset",
        help="Name of the dataset to evaluate",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name

    logger.info(f"Evaluating Opik dataset: '{dataset_name}'")

    dataset = create_dataset_from_artifacts(
        dataset_name="mathpal-testset",
        artifact_names=[
            "exam-sixth_grade-instruct-dataset"
        ],
    )
    if dataset is None:
        logger.error("Dataset can't be created. Exiting.")
        exit(1)

    experiment_config = {
        "model_id": settings.MODEL_ID,
    }
    scoring_metrics = [
        LevenshteinRatio(),
        Hallucination(),
        Moderation(),
        Style(),
    ]
    inference_pipeline = MathPal(model_id=settings.MODEL_ID)
    evaluate(
        dataset=dataset,
        task=make_evaluation_task(inference_pipeline),
        scoring_metrics=scoring_metrics,
        experiment_config=experiment_config,
    )


if __name__ == "__main__":
    main()
