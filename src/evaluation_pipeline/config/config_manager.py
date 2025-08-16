"""
Simplified configuration manager for the evaluation pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field

from ..utils import (
    ConfigurationError,
    load_yaml_config,
    save_yaml_config,
    get_device_info,
    get_environment_variables
)


class ModelConfig(BaseModel):
    """Configuration for Gemma 3N model."""
    
    name: str = Field(default="unsloth/gemma-3n-E2B-it", description="Model name or path")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    load_in_4bit: bool = Field(default=True, description="Load model in 4-bit quantization")
    load_in_8bit: bool = Field(default=False, description="Load model in 8-bit quantization")
    batch_size: int = Field(default=8, description="Batch size for inference")
    device: str = Field(default="auto", description="Device to use for inference")

class GenerationConfig(BaseModel):
    """Configuration for generation."""
    max_new_tokens: int = Field(default=512, description="Maximum number of new tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=64, description="Top-k sampling")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    
class DatasetConfig(BaseModel):
    """Configuration for dataset management."""
    source: str = Field(default="huggingface", description="Dataset source")
    dataset_id: str = Field(default="ngohongthai/exam-sixth_grade-instruct-dataset", description="Dataset ID")
    split: str = Field(default="test", description="Dataset split")
    max_samples: int = Field(default=10, description="Maximum number of samples to evaluate")
    instruction_column: str = Field(default="question", description="Instruction column")
    answer_column: str = Field(default="solution", description="Answer column")

class EvaluationSettingsConfig(BaseModel):
    """Configuration for evaluation settings."""
    save_predictions: bool = Field(default=False, description="Save model predictions")
    metrics: Dict[str, Any] = Field(
        default={
            "opik": {
                "enabled": True,
                "metrics": ["answer_relevance", "usefulness"]
            },
            "vietnamese_math": {
                "enabled": True,
                "metrics": ["mathematical_accuracy", "vietnamese_language_quality"]
            },
            "llm_as_judge": {
                "enabled": False,
                "metrics": ["accuracy", "completeness", "clarity", "relevance", "helpfulness"]
            }
        },
        description="Metrics configuration"
    )


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter integration."""
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")
    models: Dict[str, str] = Field(
        default={
            "primary": "openai/gpt-oss-20b:free",
            "fallback": "openai/gpt-oss-20b:free", 
            "judge": "openai/gpt-oss-20b:free"
        },
        description="Model configurations"
    )
    
    
class OpikConfig(BaseModel):
    """Configuration for Opik evaluation."""
    workspace: str = Field(default="mathpal", description="Opik workspace")
    project: str = Field(default="vietnamese-math-evaluation", description="Opik project")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="text", description="Log format")
    output: str = Field(default="console", description="Output: console, file, both")


class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""
    experiment_name: str = Field(description="Name of the evaluation experiment")
    output_dir: str = Field(default="./evaluation_outputs", description="Output directory")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    generation: GenerationConfig = Field(default_factory=GenerationConfig, description="Generation configuration")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig, description="Dataset configuration")
    evaluation: EvaluationSettingsConfig = Field(default_factory=EvaluationSettingsConfig, description="Evaluation configuration")
    openrouter: OpenRouterConfig = Field(description="OpenRouter configuration")
    opik: OpikConfig = Field(description="Opik configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")


class ConfigManager:
    """
    Simplified configuration manager for the evaluation pipeline.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize configuration manager."""
        self.config = config or EvaluationConfig()
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ConfigManager':
        """Create ConfigManager from YAML file."""
        try:
            raw_config = load_yaml_config(config_path)
            config = EvaluationConfig(**raw_config)
            return cls(config)
        except Exception as e:
            raise ConfigurationError(f"Error loading config from {config_path}: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """Create ConfigManager from dictionary."""
        try:
            config = EvaluationConfig(**config_dict)
            return cls(config)
        except Exception as e:
            raise ConfigurationError(f"Error creating config from dict: {e}")
    
    
    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.config.dict()
        save_yaml_config(config_dict, config_path)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config.model
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation configuration."""
        return self.config.generation
    
    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset configuration."""
        return self.config.dataset
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self.config.evaluation
    
    def get_openrouter_config(self) -> OpenRouterConfig:
        """Get OpenRouter configuration."""
        return self.config.openrouter
    
    def get_opik_config(self) -> OpikConfig:
        """Get Opik configuration."""
        return self.config.opik

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging
    
    def summary(self) -> str:
        """Get summary of configuration."""
        return f"""
        Configuration:
        - Model: {self.config.model.name}
        - Dataset: {self.config.dataset.dataset_id}
        """