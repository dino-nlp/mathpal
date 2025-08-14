"""
Simplified configuration manager for the evaluation pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator

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
    torch_dtype: str = Field(default="float16", description="Torch data type for model")
    device_map: str = Field(default="auto", description="Device mapping for model")
    use_matformer: bool = Field(default=True, description="Use MatFormer optimizations")
    
    # MatFormer configuration
    matformer_config: Dict[str, Any] = Field(
        default={
            "window_size": 512,
            "num_heads": 8,
            "use_flash_attention": True,
            "use_rope": True,
            "rope_scaling": {"type": "linear", "factor": 1.0}
        },
        description="MatFormer optimization configuration"
    )


class DatasetConfig(BaseModel):
    """Configuration for dataset management."""
    
    source: str = Field(default="huggingface", description="Dataset source")
    dataset_id: str = Field(default="ngohongthai/exam-sixth_grade-instruct-dataset", description="Dataset ID")
    split: str = Field(default="test", description="Dataset split")
    max_samples: int = Field(default=113, description="Maximum number of samples to evaluate")
    
    # Field mapping for Hugging Face datasets
    field_mapping: Dict[str, Union[str, List[str]]] = Field(
        default={
            "question": "question",
            "context": "context",
            "answer": "answer",
            "grade_level": "grade_level",
            "subject": "subject",
            "difficulty": "difficulty"
        },
        description="Mapping from internal fields to Hugging Face dataset fields"
    )


class EvaluationSettingsConfig(BaseModel):
    """Configuration for evaluation settings."""
    
    mode: str = Field(default="comprehensive", description="Evaluation mode: quick, comprehensive")
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
    
    # API key will be read from environment variable for security
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")
    
    models: Dict[str, str] = Field(
        default={
            "primary": "anthropic/claude-3.5-sonnet",
            "fallback": "openai/gpt-4o-mini", 
            "judge": "openai/gpt-4o"
        },
        description="Model configurations"
    )
    
    rate_limits: Dict[str, Any] = Field(
        default={
            "requests_per_minute": 60,
            "tokens_per_minute": 10000,
            "max_retries": 3,
            "retry_delay": 1.0
        },
        description="Rate limiting configuration"
    )


class OpikConfig(BaseModel):
    """Configuration for Opik evaluation."""
    
    # API key will be read from environment variable for security
    workspace: str = Field(default="mathpal", description="Opik workspace")
    project: str = Field(default="vietnamese-math-evaluation", description="Opik project")
    batch_size: int = Field(default=8, description="Batch size for Opik evaluation")
    max_samples: int = Field(default=113, description="Maximum samples for Opik evaluation")
    
    # Metrics to evaluate - list of metric names
    metrics: List[str] = Field(
        default=[
            "hallucination", "context_precision", "context_recall", 
            "answer_relevance", "usefulness"
        ],
        description="List of Opik metrics to evaluate"
    )
    
    # LLM-as-a-judge configuration
    llm_judge: Dict[str, Any] = Field(
        default={
            "provider": "openrouter",
            "model": "openai/gpt-4o",
            "temperature": 0.0,
            "max_tokens": 1000
        },
        description="LLM-as-a-judge configuration"
    )


class HardwareConfig(BaseModel):
    """Simplified hardware configuration."""
    
    device: str = Field(default="auto", description="Device to use: auto, cuda, cpu")
    memory_efficient: bool = Field(default=True, description="Use memory efficient settings")
    memory_fraction: float = Field(default=0.9, description="Memory fraction to use")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    optimize_for: str = Field(default="auto", description="Hardware optimization target: tesla_t4, a100, auto")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format")
    output: str = Field(default="console", description="Output: console, file, both")
    log_file: Optional[str] = Field(default=None, description="Log file path")


class PerformanceConfig(BaseModel):
    """Configuration for performance monitoring."""
    
    enable_profiling: bool = Field(default=False, description="Enable profiling")
    profile_output: str = Field(default="profiles/evaluation_profile.json", description="Profile output path")
    enable_metrics: bool = Field(default=False, description="Enable metrics collection")
    metrics_port: int = Field(default=8000, description="Metrics port")


class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""
    
    # Experiment settings
    experiment_name: str = Field(description="Name of the evaluation experiment")
    output_dir: str = Field(default="./evaluation_outputs", description="Output directory")
    
    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig, description="Dataset configuration")
    evaluation: EvaluationSettingsConfig = Field(default_factory=EvaluationSettingsConfig, description="Evaluation configuration")
    openrouter: OpenRouterConfig = Field(description="OpenRouter configuration")
    opik: OpikConfig = Field(description="Opik configuration")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, description="Hardware configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")


class ConfigManager:
    """
    Simplified configuration manager for the evaluation pipeline.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize configuration manager."""
        self.config = config or EvaluationConfig()
        self._validate_config()
    
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
    
    @classmethod
    def create_default(cls, experiment_name: str) -> 'ConfigManager':
        """Create ConfigManager with default configuration."""
        config_dict = {
            "experiment_name": experiment_name,
            "openrouter": {},
            "opik": {}
        }
        return cls.from_dict(config_dict)
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Check required environment variables
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ConfigurationError("OPENROUTER_API_KEY environment variable is required")
        if not os.getenv("OPIK_API_KEY"):
            raise ConfigurationError("OPIK_API_KEY environment variable is required")
        
        # Validate evaluation mode
        valid_modes = ["quick", "comprehensive"]
        if self.config.evaluation.mode not in valid_modes:
            raise ConfigurationError(f"Invalid evaluation mode: {self.config.evaluation.mode}")
    
    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.config.dict()
        save_yaml_config(config_dict, config_path)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config.model
    
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
    
    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware configuration."""
        return self.config.hardware
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        return self.config.performance
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get experiment information."""
        return {
            "experiment_name": self.config.experiment_name,
            "output_dir": self.config.output_dir,
            "model": self.config.model.name,
            "dataset": self.config.dataset.dataset_id,
            "evaluation_mode": self.config.evaluation.mode,
            "max_samples": self.config.dataset.max_samples
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "device_info": get_device_info(),
            "environment_variables": get_environment_variables(),
            "hardware_config": self.config.hardware.dict(),
            "performance_config": self.config.performance.dict()
        }
