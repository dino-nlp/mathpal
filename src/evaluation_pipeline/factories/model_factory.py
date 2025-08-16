"""
Model factory for the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from unsloth import FastModel, get_chat_template

from ..config import ConfigManager
from ..utils import (
    ModelError,
    get_logger,
    get_device_info
)
from ..inference import Gemma3NInferenceEngine


class Gemma3NModel:
    """
    Gemma 3N model with optimized inference.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Gemma 3N model.
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        
        # Get model and hardware config
        model_config = config.get_model_config()
        
        # Initialize inference engine
        self.inference_engine = Gemma3NInferenceEngine(model_config)
        
        self.logger.info("Gemma 3N model initialized")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load Gemma 3N model from path.
        
        Args:
            model_path: Path to model
            
        Raises:
            ModelError: If model cannot be loaded
        """
        try:
            self.inference_engine.load_model(model_path)
            self.logger.info("Gemma 3N model loaded successfully")
            
        except Exception as e:
            raise ModelError(f"Error loading Gemma 3N model: {e}")
    

    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response for given prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Raises:
            ModelError: If generation fails
        """
        try:
            return self.inference_engine.generate(question=prompt, **kwargs)
        except Exception as e:
            raise ModelError(f"Error generating response: {e}")
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
            
        Raises:
            ModelError: If generation fails
        """
        try:
            return self.inference_engine.generate_batch(questions=prompts, **kwargs)
        except Exception as e:
            raise ModelError(f"Error in batch generation: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information
        """
        # Return basic model info since we don't have stats anymore
        return {
            "model_type": "Gemma3N",
            "device": str(self.device),
            "model_loaded": self.inference_engine.model is not None
        }


