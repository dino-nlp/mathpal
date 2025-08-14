"""
Evaluation manager for the evaluation pipeline.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from ..config import ConfigManager
from ..utils import (
    EvaluationPipelineError,
    get_logger,
    create_output_directory,
    save_evaluation_results
)


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    
    experiment_name: str
    model_path: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    samples_evaluated: int
    evaluation_time: float
    output_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "samples_evaluated": self.samples_evaluated,
            "evaluation_time": self.evaluation_time,
            "output_path": self.output_path
        }


class BaseEvaluationManager(ABC):
    """
    Base class for evaluation managers.
    
    Provides common functionality for managing evaluation processes.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize evaluation manager.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Setup logging
        logging_config = config.get_logging_config()
        if logging_config.log_file:
            self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Create output directory
        self.output_dir = create_output_directory(
            config.config.output_dir,
            config.config.experiment_name
        )
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    @abstractmethod
    def evaluate_model(self, model_path: Union[str, Path]) -> EvaluationResult:
        """
        Evaluate a model.
        
        Args:
            model_path: Path to the model to evaluate
            
        Returns:
            Evaluation result
        """
        pass
    
    @abstractmethod
    def evaluate_dataset(self, dataset_path: Union[str, Path]) -> EvaluationResult:
        """
        Evaluate a dataset.
        
        Args:
            dataset_path: Path to the dataset to evaluate
            
        Returns:
            Evaluation result
        """
        pass
    
    def save_results(self, results: EvaluationResult) -> Path:
        """
        Save evaluation results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Path to saved results file
        """
        results_dict = results.to_dict()
        file_path = save_evaluation_results(
            results_dict,
            self.output_dir,
            f"{results.experiment_name}_results.json"
        )
        
        self.logger.info(f"Saved results to: {file_path}")
        return file_path
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get experiment information.
        
        Returns:
            Experiment information
        """
        return self.config.get_experiment_info()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            System information
        """
        return self.config.get_system_info()


class EvaluationManager(BaseEvaluationManager):
    """
    Main evaluation manager for the evaluation pipeline.
    
    Orchestrates the evaluation process using Gemma 3N, Opik, and OpenRouter.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize evaluation manager.
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        
        # Initialize sub-managers
        from .dataset_manager import DatasetManager
        from .metrics_manager import MetricsManager
        
        self.dataset_manager = DatasetManager(self.config)
        self.metrics_manager = MetricsManager(self.config)
        
        self.logger.info("Evaluation manager initialized")
    
    def evaluate_model(
        self, 
        model_path: Union[str, Path],
        dataset_path: Optional[Union[str, Path]] = None,
        samples: Optional[List] = None
    ) -> EvaluationResult:
        """
        Evaluate a model.
        
        Args:
            model_path: Path to the model to evaluate
            dataset_path: Optional path to custom dataset
            samples: Optional pre-loaded dataset samples
            
        Returns:
            Evaluation result
        """
        import time
        
        start_time = time.time()
        
        self.logger.info(f"Starting evaluation of model: {model_path}")
        
        # Validate model path (support both local paths and Hugging Face model names)
        if "/" in str(model_path) and not Path(model_path).exists():
            # This might be a Hugging Face model name, skip local path validation
            self.logger.info(f"Using Hugging Face model: {model_path}")
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise EvaluationPipelineError(f"Model path does not exist: {model_path}")
        
        # Managers are already initialized in __init__
        
        # Use pre-loaded samples if provided, otherwise load dataset
        if samples is not None:
            dataset = samples
        elif dataset_path:
            dataset = self.dataset_manager.load_dataset(dataset_path)
        else:
            dataset = self.dataset_manager.get_default_dataset()
        
        # Apply max_samples limit from config
        max_samples = self.config.config.dataset.max_samples
        if max_samples and len(dataset) > max_samples:
            dataset = dataset[:max_samples]
            self.logger.info(f"Limited evaluation to {max_samples} samples")
        
        # Run evaluation
        metrics = self.metrics_manager.evaluate_model_on_dataset(model_path, dataset)
        
        # Calculate evaluation time
        evaluation_time = time.time() - start_time
        
        # Create result
        result = EvaluationResult(
            experiment_name=self.config.config.experiment_name,
            model_path=str(model_path),
            metrics=metrics,
            metadata={
                "config": self.config.get_experiment_info(),
                "system_info": self.get_system_info(),
                "dataset_info": self.dataset_manager.get_dataset_info()
            },
            samples_evaluated=len(dataset),
            evaluation_time=evaluation_time,
            output_path=str(self.output_dir)
        )
        
        # Save results
        self.save_results(result)
        
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        self.logger.info(f"Evaluated {len(dataset)} samples")
        
        return result
    
    def evaluate_dataset(self, dataset_path: Union[str, Path]) -> EvaluationResult:
        """
        Evaluate a dataset (placeholder for future implementation).
        
        Args:
            dataset_path: Path to the dataset to evaluate
            
        Returns:
            Evaluation result
        """
        # This method would be used for dataset-specific evaluation
        # For now, we'll use the model evaluation with the specified dataset
        raise NotImplementedError("Dataset evaluation not yet implemented")
    
    def run_quick_evaluation(
        self, 
        model_path: Union[str, Path], 
        num_samples: int = 10
    ) -> EvaluationResult:
        """
        Run a quick evaluation with limited samples.
        
        Args:
            model_path: Path to the model to evaluate
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation result
        """
        self.logger.info(f"Running quick evaluation with {num_samples} samples")
        
        # Temporarily update config for quick evaluation
        original_max_samples = self.config.config.opik.max_samples
        self.config.config.opik.max_samples = num_samples
        
        try:
            result = self.evaluate_model(model_path)
            return result
        finally:
            # Restore original config
            self.config.config.opik.max_samples = original_max_samples
    
    def run_comprehensive_evaluation(
        self, 
        model_path: Union[str, Path]
    ) -> EvaluationResult:
        """
        Run a comprehensive evaluation with all metrics.
        
        Args:
            model_path: Path to the model to evaluate
            
        Returns:
            Evaluation result
        """
        self.logger.info("Running comprehensive evaluation")
        
        # Ensure all metrics are enabled
        all_metrics = [
            "hallucination", "context_precision", "context_recall",
            "answer_relevance", "usefulness", "moderation",
            "conversational_coherence", "session_completeness_quality",
            "user_frustration", "mathematical_accuracy",
            "vietnamese_language_quality", "step_by_step_reasoning",
            "grade_level_appropriateness", "problem_solving_approach"
        ]
        
        original_metrics = self.config.config.opik.metrics
        self.config.config.opik.metrics = all_metrics
        
        try:
            result = self.evaluate_model(model_path)
            return result
        finally:
            # Restore original metrics
            self.config.config.opik.metrics = original_metrics
    
    def cleanup(self):
        """
        Cleanup resources and free memory.
        
        This method should be called when the evaluation manager is no longer needed
        to ensure proper resource cleanup.
        """
        try:
            self.logger.info("Cleaning up evaluation manager resources...")
            
            # Cleanup sub-managers
            if hasattr(self, 'dataset_manager') and self.dataset_manager is not None:
                if hasattr(self.dataset_manager, 'cleanup'):
                    self.dataset_manager.cleanup()
            
            if hasattr(self, 'metrics_manager') and self.metrics_manager is not None:
                if hasattr(self.metrics_manager, 'cleanup'):
                    self.metrics_manager.cleanup()
            
            # Clear references
            self.dataset_manager = None
            self.metrics_manager = None
            
            self.logger.info("Evaluation manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def __del__(self):
        """
        Destructor to ensure cleanup is called.
        """
        try:
            self.cleanup()
        except Exception:
            # Ignore errors in destructor
            pass
