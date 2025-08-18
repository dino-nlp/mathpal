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



class EvaluationManager:
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
        # Initialize sub-managers
        from .metrics_manager import MetricsManager
        self.config = config
        self.metrics_manager = MetricsManager(config)
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.logger.info("Evaluation manager initialized")
    
    def evaluate_model(self, samples: Optional[List] = None) -> EvaluationResult:
        """
        Evaluate a model.
        
        Args:
            samples: Optional pre-loaded dataset samples with predictions
            
        Returns:
            Evaluation result
        """
        import time
        
        start_time = time.time()
        
        self.logger.info(f"Starting evaluation of {len(samples)} samples")
        
        # Run evaluation
        metrics = self.metrics_manager.evaluate_model_on_dataset(samples)
        
        # Calculate evaluation time
        evaluation_time = time.time() - start_time
        
        # Create result
        result = EvaluationResult(
            experiment_name=self.config.get_experiment_config().experiment_name,
            model_path=self.config.get_model_config().name,
            metrics=metrics,
            metadata={
                "config": self.config.get_experiment_info(),
                "system_info": self.get_system_info(),
            },
            samples_evaluated=len(samples),
            evaluation_time=evaluation_time,
            output_path=str(self.config.get_output_config().output_dir)
        )
        
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        self.logger.info(f"Evaluated {len(samples)} samples")
        
        return result
    
    def save_results(self, result: EvaluationResult) -> None:
        """
        Save evaluation results to file.
        
        Args:
            result: Evaluation result to save
        """
        try:
            output_dir = Path(self.config.get_output_config().output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results to JSON file
            output_file = output_dir / f"{result.experiment_name}_results.json"
            save_evaluation_results(result.to_dict(), output_file)
            
            self.logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}", exc_info=True)
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for metadata.
        
        Returns:
            System information dictionary
        """
        import platform
        import torch
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    
    def cleanup(self):
        """
        Cleanup resources and free memory.
        
        This method should be called when the evaluation manager is no longer needed
        to ensure proper resource cleanup.
        """
        try:
            self.logger.info("Cleaning up evaluation manager resources...")
             
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
