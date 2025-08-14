"""
Model factory for the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch

from ..config import ConfigManager
from ..utils import (
    ModelError,
    get_logger,
    get_device_info
)
from ..inference import Gemma3NInferenceEngine


class BaseModel(ABC):
    """
    Base class for all models in the evaluation pipeline.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize base model.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """
        Setup device for model inference.
        
        Returns:
            Device to use for inference
        """
        device_info = get_device_info()
        
        if device_info["cuda_available"]:
            device = torch.device("cuda")
            self.logger.info(f"Using CUDA device: {device_info['device_name']}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU device")
        
        return device
    
    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model from path.
        
        Args:
            model_path: Path to model
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response for given prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        pass


class Gemma3NModel(BaseModel):
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
        hardware_config = config.get_hardware_config()
        
        # Create ModelConfig dataclass
        from ..inference.gemma3n_inference import ModelConfig
        model_config_obj = ModelConfig(
            name=model_config.name,
            max_seq_length=model_config.max_seq_length,
            load_in_4bit=model_config.load_in_4bit,
            load_in_8bit=model_config.load_in_8bit,
            batch_size=model_config.batch_size,
            torch_dtype=model_config.torch_dtype,
            device_map=model_config.device_map
        )
        
        # Initialize inference engine
        self.inference_engine = Gemma3NInferenceEngine(model_config_obj, hardware_config)
        
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


class ModelFactory:
    """
    Factory for creating model instances.
    """
    
    @staticmethod
    def create_model(config: ConfigManager, model_type: str = "gemma3n") -> BaseModel:
        """
        Create a model instance.
        
        Args:
            config: Configuration manager
            model_type: Type of model to create
            
        Returns:
            Model instance
            
        Raises:
            ModelError: If model type is not supported
        """
        if model_type.lower() == "gemma3n":
            return Gemma3NModel(config)
        else:
            raise ModelError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_supported_models() -> list:
        """
        Get list of supported model types.
        
        Returns:
            List of supported model types
        """
        return ["gemma3n"]
