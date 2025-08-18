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
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.logger.info("Evaluation manager initialized")
    
    def evaluate_model(self, samples: Optional[List] = None):
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
        self.logger.info(f"Samples: {samples}")
        
        # Calculate evaluation time
        evaluation_time = time.time() - start_time
        
        
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        self.logger.info(f"Evaluated {len(samples)} samples")
        
        return result
    
    # def save_results(self, result: EvaluationResult) -> None:
    #     """
    #     Save evaluation results to file.
        
    #     Args:
    #         result: Evaluation result to save
    #     """
    #     try:
    #         output_dir = Path(self.config.get_output_config().output_dir)
    #         output_dir.mkdir(parents=True, exist_ok=True)
            
    #         # Save results to JSON file
    #         output_file = output_dir / f"{result.experiment_name}_results.json"
    #         save_evaluation_results(result.to_dict(), output_file)
            
    #         self.logger.info(f"Results saved to: {output_file}")
            
    #     except Exception as e:
    #         self.logger.error(f"Error saving results: {e}", exc_info=True)
    #         raise
    
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
    