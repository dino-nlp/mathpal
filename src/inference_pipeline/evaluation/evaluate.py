import argparse
import os
import torch
from typing import Dict, Any, Optional
from tqdm import tqdm
import time

from config import settings
from core.logger_utils import get_logger
from core.opik_utils import create_dataset_from_artifacts
from mathpal import MathPal
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, LevenshteinRatio, Moderation

from .style import Style
from .progress_metrics import (
    create_progress_metrics, 
    setup_metric_progress_bars, 
    cleanup_metric_progress_bars
)

# Disable tokenizers parallelism to avoid deadlocks during forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure TorchDynamo to prevent recompile limit issues during evaluation
# Use a more conservative compilation strategy for evaluation
torch._dynamo.config.cache_size_limit = 256  # Increase cache size limit significantly
torch._dynamo.config.suppress_errors = True   # Suppress compilation errors

logger = get_logger(__name__)


class EvaluationProgressCallback:
    """
    Custom callback class to track evaluation progress and display it with tqdm.
    """
    
    def __init__(self, total_samples: int, experiment_name: str = "Evaluation"):
        self.total_samples = total_samples
        self.experiment_name = experiment_name
        self.current_sample = 0
        self.start_time = None
        self.progress_bar = None
        self.metrics_progress = {}
        
    def on_evaluation_start(self):
        """Called when evaluation starts"""
        self.start_time = time.time()
        logger.info(f"🚀 Starting {self.experiment_name} evaluation with {self.total_samples} samples")
        
        # Create progress bar
        self.progress_bar = tqdm(
            total=self.total_samples,
            desc=f"📊 {self.experiment_name}",
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
    def on_sample_start(self, sample_idx: int, sample_data: Dict[str, Any]):
        """Called when processing a sample starts"""
        self.current_sample = sample_idx
        if self.progress_bar:
            self.progress_bar.set_postfix({
                "Sample": f"{sample_idx + 1}/{self.total_samples}",
                "Question": sample_data.get("question", "")[:30] + "..." if len(sample_data.get("question", "")) > 30 else sample_data.get("question", "")
            })
    
    def on_sample_complete(self, sample_idx: int, result: Dict[str, Any]):
        """Called when a sample evaluation is complete"""
        if self.progress_bar:
            self.progress_bar.update(1)
            
        # Log sample completion with metrics
        metrics_summary = {}
        for key, value in result.items():
            if isinstance(value, (int, float)):
                metrics_summary[key] = f"{value:.3f}"
            elif isinstance(value, dict) and "value" in value:
                metrics_summary[key] = f"{value['value']:.3f}"
        
        logger.debug(f"✅ Sample {sample_idx + 1}/{self.total_samples} completed - Metrics: {metrics_summary}")
    
    def on_metric_start(self, metric_name: str):
        """Called when a metric evaluation starts"""
        if metric_name not in self.metrics_progress:
            self.metrics_progress[metric_name] = 0
        self.metrics_progress[metric_name] += 1
        
        if self.progress_bar:
            self.progress_bar.set_postfix({
                "Current Metric": metric_name,
                "Metric Progress": f"{self.metrics_progress[metric_name]}/{self.total_samples}"
            })
    
    def on_evaluation_complete(self, results: Dict[str, Any]):
        """Called when evaluation is complete"""
        if self.progress_bar:
            self.progress_bar.close()
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_time_per_sample = elapsed_time / self.total_samples if self.total_samples > 0 else 0
        
        logger.info(f"🎉 {self.experiment_name} completed!")
        logger.info(f"⏱️  Total time: {elapsed_time:.2f}s")
        logger.info(f"📈 Average time per sample: {avg_time_per_sample:.2f}s")
        logger.info(f"📊 Total samples evaluated: {self.total_samples}")
        
        # Log summary metrics
        if results and "metrics" in results:
            logger.info("📋 Evaluation Results Summary:")
            for metric_name, metric_value in results["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    logger.info(f"   {metric_name}: {metric_value:.4f}")
                elif isinstance(metric_value, dict):
                    logger.info(f"   {metric_name}: {metric_value}")


def make_evaluation_task(inference_pipeline: MathPal, progress_callback: Optional[EvaluationProgressCallback] = None):
    def evaluation_task(x: dict) -> dict:
        # Notify progress callback
        if progress_callback:
            progress_callback.on_sample_start(x.get("_sample_idx", 0), x)
        
        # Generate answer
        answer = inference_pipeline.generate(
            question=x["question"],
        )["answer"]

        result = {
            "input": x["question"],
            "output": answer,
            "expected_output": x["solution"],
            "reference": x["solution"],
        }
        
        # Notify progress callback
        if progress_callback:
            progress_callback.on_sample_complete(x.get("_sample_idx", 0), result)
        
        return result

    return evaluation_task


def get_scoring_metrics(use_progress_metrics: bool = False, track_progress: bool = True):
    """
    Get scoring metrics based on configuration.
    
    Args:
        use_progress_metrics: Whether to use custom progress metrics
        track_progress: Whether to enable progress tracking
        
    Returns:
        List of metric instances
    """
    if use_progress_metrics:
        logger.info("📊 Using custom progress metrics with detailed tracking")
        metrics = create_progress_metrics(track_progress=track_progress)
        # Add Style metric (which doesn't have progress tracking)
        metrics.append(Style())
        return metrics
    else:
        logger.info("📊 Using standard Opik metrics")
        return [
            LevenshteinRatio(),
            Hallucination(model=settings.OPENROUTER_BASE_MODEL),
            Moderation(model=settings.OPENROUTER_BASE_MODEL),
            Style(),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate monitoring script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mathpal-testset",
        help="Name of the dataset to evaluate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="MathPal Evaluation",
        help="Name of the evaluation experiment",
    )
    parser.add_argument(
        "--use_progress_metrics",
        action="store_true",
        help="Use custom progress metrics with detailed tracking",
    )
    parser.add_argument(
        "--no_progress_tracking",
        action="store_true",
        help="Disable progress tracking (faster but less informative)",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    max_samples = args.max_samples
    experiment_name = args.experiment_name
    use_progress_metrics = args.use_progress_metrics
    track_progress = not args.no_progress_tracking

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

    # Get total number of samples
    total_samples = len(dataset.get_items())
    if max_samples:
        total_samples = min(total_samples, max_samples)
        logger.info(f"Limiting evaluation to {max_samples} samples")

    # Create progress callback
    progress_callback = EvaluationProgressCallback(
        total_samples=total_samples,
        experiment_name=experiment_name
    ) if track_progress else None

    # Get scoring metrics
    scoring_metrics = get_scoring_metrics(
        use_progress_metrics=use_progress_metrics,
        track_progress=track_progress
    )

    experiment_config = {
        "model_id": settings.MODEL_ID,
        "experiment_name": experiment_name,
        "max_samples": max_samples,
        "use_progress_metrics": use_progress_metrics,
        "track_progress": track_progress,
    }
    
    inference_pipeline = MathPal(model_id=settings.MODEL_ID)
    
    # Setup metric progress bars if using progress metrics
    if use_progress_metrics and track_progress:
        setup_metric_progress_bars(scoring_metrics, total_samples)
    
    # Start progress tracking
    if progress_callback:
        progress_callback.on_evaluation_start()
    
    try:
        # Run evaluation with progress tracking
        results = evaluate(
            dataset=dataset,
            task=make_evaluation_task(inference_pipeline, progress_callback),
            scoring_metrics=scoring_metrics,
            experiment_config=experiment_config,
            nb_samples=max_samples,
        )
        
        # Complete progress tracking
        if progress_callback:
            progress_callback.on_evaluation_complete(results)
        
        # Cleanup metric progress bars
        if use_progress_metrics and track_progress:
            cleanup_metric_progress_bars(scoring_metrics)
        
        # Log final results
        logger.info("📊 Final Evaluation Results:")
        if hasattr(results, 'metrics'):
            for metric_name, metric_value in results.metrics.items():
                logger.info(f"   {metric_name}: {metric_value}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        if progress_callback and progress_callback.progress_bar:
            progress_callback.progress_bar.close()
        if use_progress_metrics and track_progress:
            cleanup_metric_progress_bars(scoring_metrics)
        raise


if __name__ == "__main__":
    main()
