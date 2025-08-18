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
        self.model_config = config.get_model_config()
        self.logger.info("Gemma 3N model initialized")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load Gemma 3N model from path.
        
        Args:
            model_path: Path to model
            
        Raises:
            ModelError: If model cannot be loaded
        """
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_config.name,
            dtype=None,  # Auto-detect
            max_seq_length=self.model_config.max_seq_length,
            load_in_4bit=self.model_config.load_in_4bit,
            load_in_8bit=self.model_config.load_in_8bit,
        )
        self.tokenizer = get_chat_template(self.tokenizer, "gemma-3n")
        FastModel.for_inference(self.model)
        self.model.to(self.model_config.device)
        self.logger.info("Model loaded successfully")
        return self.model, self.tokenizer
    

    

