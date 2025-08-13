"""
Centralized Configuration Management System
==========================================

This module provides a unified approach to managing configurations across the training pipeline.
It solves the inconsistency issues between different config formats and provides type-safe 
access to configuration sections.

Key Features:
- Unified config loading from multiple YAML formats
- Type-safe config section access
- Environment variable and CLI override support
- Validation and error handling
- Dependency injection for managers
"""

import os
import yaml
from typing import Dict, Any, Optional, Type, TypeVar, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from training_pipeline.utils.exceptions import ConfigurationError, ValidationError

# Type variable for config sections
T = TypeVar('T', bound='ConfigSection')

class ConfigSection(ABC):
    """Base class for configuration sections."""
    
    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config section from dictionary."""
        pass
    
    @abstractmethod
    def validate(self) -> None:
        """Validate configuration section."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()


@dataclass
class ModelConfigSection(ConfigSection):
    """Model configuration section."""
    name: str = "unsloth/gemma-3n-E4B-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfigSection':
        # Handle both nested and flat formats
        if 'model' in data:
            model_data = data['model']
        else:
            # Flat format - extract model_* fields
            model_data = {
                'name': data.get('model_name', cls.name),
                'max_seq_length': data.get('max_seq_length', cls.max_seq_length),
                'load_in_4bit': data.get('load_in_4bit', cls.load_in_4bit),
                'load_in_8bit': data.get('load_in_8bit', cls.load_in_8bit),
                'full_finetuning': data.get('full_finetuning', cls.full_finetuning),
            }
        
        return cls(
            name=model_data.get('name', cls.name),
            max_seq_length=model_data.get('max_seq_length', cls.max_seq_length),
            load_in_4bit=model_data.get('load_in_4bit', cls.load_in_4bit),
            load_in_8bit=model_data.get('load_in_8bit', cls.load_in_8bit),
            full_finetuning=model_data.get('full_finetuning', cls.full_finetuning),
        )
    
    def validate(self) -> None:
        if not self.name:
            raise ValidationError("Model name cannot be empty")
        if self.max_seq_length <= 0:
            raise ValidationError("max_seq_length must be positive")
        if self.load_in_4bit and self.load_in_8bit:
            raise ValidationError("Cannot use both 4-bit and 8-bit quantization")


@dataclass
class DatasetConfigSection(ConfigSection):
    """Dataset configuration section."""
    name: str = "ngohongthai/exam-sixth_grade-instruct-dataset"
    train_split: str = "train"
    test_split: str = "test"
    text_field: str = "text"
    instruction_column: str = "question"
    answer_column: str = "solution"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetConfigSection':
        if 'dataset' in data:
            dataset_data = data['dataset']
        else:
            # Flat format
            dataset_data = {
                'name': data.get('dataset_name', cls.name),
                'train_split': data.get('train_split', cls.train_split),
                'test_split': data.get('test_split', cls.test_split),
                'text_field': data.get('dataset_text_field', data.get('text_field', cls.text_field)),
                'instruction_column': data.get('instruction_column', cls.instruction_column),
                'answer_column': data.get('answer_column', cls.answer_column)
            }
        
        return cls(
            name=dataset_data.get('name', cls.name),
            train_split=dataset_data.get('train_split', cls.train_split),
            test_split=dataset_data.get('test_split', cls.test_split),
            text_field=dataset_data.get('text_field', cls.text_field),
            instruction_column=dataset_data.get('instruction_column', cls.instruction_column),
            answer_column=dataset_data.get('answer_column', cls.num_answer_columnproc)
        )
    
    def validate(self) -> None:
        if not self.name:
            raise ValidationError("Dataset name cannot be empty")
        if not self.text_field:
            raise ValidationError("Text field cannot be empty")


@dataclass
class TrainingConfigSection(ConfigSection):
    """Training configuration section."""
    max_steps: int = 100
    num_train_epochs: Optional[int] = None
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    train_on_responses_only: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfigSection':
        if 'training' in data:
            training_data = data['training']
        else:
            # Flat format
            training_data = {k: v for k, v in data.items() 
                           if k in cls.__dataclass_fields__}
        
        return cls(
            max_steps=training_data.get('max_steps', cls.max_steps),
            num_train_epochs=training_data.get('num_train_epochs', cls.num_train_epochs),
            per_device_train_batch_size=training_data.get('per_device_train_batch_size', cls.per_device_train_batch_size),
            per_device_eval_batch_size=training_data.get('per_device_eval_batch_size', cls.per_device_eval_batch_size),
            gradient_accumulation_steps=training_data.get('gradient_accumulation_steps', cls.gradient_accumulation_steps),
            learning_rate=training_data.get('learning_rate', cls.learning_rate),
            warmup_ratio=training_data.get('warmup_ratio', cls.warmup_ratio),
            weight_decay=training_data.get('weight_decay', cls.weight_decay),
            optim=training_data.get('optim', cls.optim),
            lr_scheduler_type=training_data.get('lr_scheduler_type', cls.lr_scheduler_type),
            max_grad_norm=training_data.get('max_grad_norm', cls.max_grad_norm),
            fp16=training_data.get('fp16', cls.fp16),
            bf16=training_data.get('bf16', cls.bf16),
            train_on_responses_only=training_data.get('train_on_responses_only', cls.train_on_responses_only),
        )
    
    def validate(self) -> None:
        if self.max_steps <= 0 and (self.num_train_epochs is None or self.num_train_epochs <= 0):
            raise ValidationError("Must specify either max_steps > 0 or num_train_epochs > 0")
        if self.learning_rate <= 0:
            raise ValidationError("learning_rate must be positive")


@dataclass
class LoRAConfigSection(ConfigSection):
    """LoRA configuration section."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_rslora: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoRAConfigSection':
        if 'lora' in data:
            lora_data = data['lora']
        else:
            # Flat format with lora_ prefix
            default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            lora_data = {
                'r': data.get('lora_r', cls.r),
                'alpha': data.get('lora_alpha', cls.alpha),
                'dropout': data.get('lora_dropout', cls.dropout),
                'bias': data.get('lora_bias', cls.bias),
                'target_modules': data.get('lora_target_modules', data.get('target_modules', default_target_modules)),
                'use_rslora': data.get('use_rslora', cls.use_rslora),
            }
        
        default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return cls(
            r=lora_data.get('r', cls.r),
            alpha=lora_data.get('alpha', cls.alpha),
            dropout=lora_data.get('dropout', cls.dropout),
            bias=lora_data.get('bias', cls.bias),
            target_modules=lora_data.get('target_modules', default_target_modules),
            use_rslora=lora_data.get('use_rslora', cls.use_rslora),
        )
    
    def validate(self) -> None:
        if self.r <= 0:
            raise ValidationError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValidationError("LoRA alpha must be positive")


@dataclass
class OutputConfigSection(ConfigSection):
    """Output configuration section."""
    base_dir: str = "outputs"
    experiment_name: str = "gemma3n-experiment"
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    save_formats: List[str] = field(default_factory=lambda: ["lora", "merged_16bit"])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputConfigSection':
        if 'output' in data:
            output_data = data['output']
        else:
            # Flat format
            default_save_formats = ["lora", "merged_16bit"]
            output_data = {
                'base_dir': data.get('output_dir', cls.base_dir),
                'experiment_name': data.get('experiment_name', cls.experiment_name),
                'save_strategy': data.get('save_strategy', cls.save_strategy),
                'save_steps': data.get('save_steps', cls.save_steps),
                'save_total_limit': data.get('save_total_limit', cls.save_total_limit),
                'load_best_model_at_end': data.get('load_best_model_at_end', cls.load_best_model_at_end),
                'save_formats': data.get('save_formats', default_save_formats),
            }
        
        default_save_formats = ["lora", "merged_16bit"]
        return cls(
            base_dir=output_data.get('base_dir', cls.base_dir),
            experiment_name=output_data.get('experiment_name', cls.experiment_name),
            save_strategy=output_data.get('save_strategy', cls.save_strategy),
            save_steps=output_data.get('save_steps', cls.save_steps),
            save_total_limit=output_data.get('save_total_limit', cls.save_total_limit),
            load_best_model_at_end=output_data.get('load_best_model_at_end', cls.load_best_model_at_end),
            save_formats=output_data.get('save_formats', default_save_formats),
        )
    
    def validate(self) -> None:
        if not self.base_dir:
            raise ValidationError("Output base directory cannot be empty")
        if not self.experiment_name:
            raise ValidationError("Experiment name cannot be empty")
    
    def get_output_dir(self) -> str:
        """Get full output directory path."""
        return f"{self.base_dir}/{self.experiment_name}"

@dataclass
class LoggingConfigSection(ConfigSection):
    logging_steps: int = 5
    report_to: List[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfigSection':
        if 'logging' in data:
            logging_data = data['logging']
        else:
            logging_data = {
                'logging_steps': 5,
                'report_to': None
            }
        
        return cls(
            logging_steps=logging_data.get('logging_steps', 5),
            report_to=logging_data.get('report_to', None)
        )
    
    def validate(self) -> None:
        pass  # No specific validation needed


@dataclass
class HubConfigSection(ConfigSection):
    push_to_hub: bool = False
    username: str = None 
    repo_name: str = None 
    private: bool = False 
    token: str = None 
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HubConfigSection':
        if 'hub' in data:
            hub_data = data['hub']
        else:
            hub_data = {
                'push_to_hub': False,
                'username': None,
                'repo_name': None,
                'private': False
            }
        
        hub_data['token'] = os.getenv('HF_TOKEN', default=None)
        
        return cls(
            push_to_hub=hub_data.get('push_to_hub', False),
            username=hub_data.get('username', None),
            repo_name=hub_data.get('repo_name', None),
            private=hub_data.get('private', False),
            token=hub_data.get('token', None)
        )
    
    def validate(self) -> None:
        if not self.push_to_hub:
            pass
        else:
            if (not self.username) or (not self.repo_name) or (not self.token):
                raise ValidationError("Username or repo name or HF_TOKEN is empty")

@dataclass
class EvaluationConfigSection(ConfigSection):
    regression_test: bool = False
    num_tc_examples: int = 2

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfigSection':
        if 'evaluation' in data:
            evaluation_data = data['evaluation']
        else:
            evaluation_data = {
                "regression_test": False,
                "num_tc_examples": 2
            }
        
        return cls(
            regression_test=evaluation_data.get('regression_test', False),
            num_tc_examples=evaluation_data.get('num_tc_examples', 2)
        )

    def validate(self) -> None:
        pass
        

@dataclass
class GenerationConfigSection(ConfigSection):
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    do_sample: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationConfigSection':
        if 'generation' in data:
            generation_data = data['generation']
        else:
            generation_data = {
                "max_new_tokens": 512,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "do_sample": True
            }
        return cls(
            max_new_tokens=generation_data.get('max_new_tokens', 512),
            temperature=generation_data.get('temperature', 1.0),
            top_p=generation_data.get('top_p', 0.95),
            top_k=generation_data.get('top_k', 64),
            do_sample=generation_data.get('do_sample', True)
        )
    
    def validate(self) -> None:
        pass


@dataclass
class CometConfigSection(ConfigSection):
    """Comet ML configuration section."""
    experiment_name: str = "gemma3n-experiment"
    tags: List[str] = field(default_factory=lambda: ["gemma3n", "vietnamese", "math"])
    auto_metric_logging: bool = True
    auto_param_logging: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CometConfigSection':
        if 'comet' in data:
            comet_data = data['comet']
        else:
            # Handle flat format
            default_tags = ["gemma3n", "vietnamese", "math"]
            comet_data = {
                'experiment_name': data.get('experiment_name', cls.experiment_name),
                'tags': data.get('tags', default_tags),
                'auto_metric_logging': data.get('auto_metric_logging', cls.auto_metric_logging),
                'auto_param_logging': data.get('auto_param_logging', cls.auto_param_logging),
            }
        
        default_tags = ["gemma3n", "vietnamese", "math"]
        return cls(
            experiment_name=comet_data.get('experiment_name', cls.experiment_name),
            tags=comet_data.get('tags', default_tags),
            auto_metric_logging=comet_data.get('auto_metric_logging', cls.auto_metric_logging),
            auto_param_logging=comet_data.get('auto_param_logging', cls.auto_param_logging),
        )
    
    def validate(self) -> None:
        pass  # No specific validation needed


@dataclass
class SystemConfigSection(ConfigSection):
    """System configuration section."""
    seed: int = 42
    use_gradient_checkpointing: Union[str, bool] = "unsloth"
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfigSection':
        if 'system' in data:
            system_data = data['system']
        else:
            # Flat format
            system_data = {
                'seed': data.get('seed', cls.seed),
                'use_gradient_checkpointing': data.get('use_gradient_checkpointing', cls.use_gradient_checkpointing),
                'dataloader_drop_last': data.get('dataloader_drop_last', cls.dataloader_drop_last),
                'dataloader_pin_memory': data.get('dataloader_pin_memory', cls.dataloader_pin_memory),
                'dataloader_num_workers': data.get('dataloader_num_workers', cls.dataloader_num_workers),
            }
        
        return cls(
            seed=system_data.get('seed', cls.seed),
            use_gradient_checkpointing=system_data.get('use_gradient_checkpointing', cls.use_gradient_checkpointing),
            dataloader_drop_last=system_data.get('dataloader_drop_last', cls.dataloader_drop_last),
            dataloader_pin_memory=system_data.get('dataloader_pin_memory', cls.dataloader_pin_memory),
            dataloader_num_workers=system_data.get('dataloader_num_workers', cls.dataloader_num_workers),
        )
    
    def validate(self) -> None:
        pass  # No specific validation needed


class ConfigManager:
    """
    Centralized configuration manager that handles loading, validation, 
    and providing type-safe access to configuration sections.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to YAML config file
            config_data: Direct config data dictionary
        """
        self.config_path = config_path
        self.raw_config = config_data or {}
        
        if config_path:
            self.load_from_file(config_path)
        
        # Initialize config sections
        self._model_config = None
        self._dataset_config = None
        self._training_config = None
        self._lora_config = None
        self._output_config = None
        self._comet_config = None
        self._system_config = None
        self._logging_config = None
        self._hub_config = None 
        self._evaluation_config = None
        self._generation_config = None
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.raw_config = yaml.safe_load(f) or {}
            self.config_path = config_path
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_path}: {e}")
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides."""
        self.raw_config.update(overrides)
        # Reset cached config sections
        self._reset_cached_configs()
    
    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_overrides = {}
        
        # Model overrides
        if os.getenv('MODEL_NAME'):
            env_overrides['model_name'] = os.getenv('MODEL_NAME')
        
        # Dataset overrides
        if os.getenv('DATASET_NAME'):
            env_overrides['dataset_name'] = os.getenv('DATASET_NAME')
        
        # Training overrides
        if os.getenv('MAX_STEPS'):
            env_overrides['max_steps'] = int(os.getenv('MAX_STEPS'))
        
        if os.getenv('LEARNING_RATE'):
            env_overrides['learning_rate'] = float(os.getenv('LEARNING_RATE'))
        
        # Output overrides
        if os.getenv('OUTPUT_DIR'):
            env_overrides['output_dir'] = os.getenv('OUTPUT_DIR')
        
        if os.getenv('EXPERIMENT_NAME'):
            env_overrides['experiment_name'] = os.getenv('EXPERIMENT_NAME')
        
        if env_overrides:
            self.apply_overrides(env_overrides)
    
    def _reset_cached_configs(self) -> None:
        """Reset cached configuration sections."""
        self._model_config = None
        self._dataset_config = None
        self._training_config = None
        self._lora_config = None
        self._output_config = None
        self._comet_config = None
        self._system_config = None
        self._logging_config = None
        self._hub_config = None 
        self._evaluation_config = None
        self._generation_config = None
    
    @property
    def model(self) -> ModelConfigSection:
        """Get model configuration section."""
        if self._model_config is None:
            self._model_config = ModelConfigSection.from_dict(self.raw_config)
            self._model_config.validate()
        return self._model_config
    
    @property
    def dataset(self) -> DatasetConfigSection:
        """Get dataset configuration section."""
        if self._dataset_config is None:
            self._dataset_config = DatasetConfigSection.from_dict(self.raw_config)
            self._dataset_config.validate()
        return self._dataset_config
    
    @property
    def training(self) -> TrainingConfigSection:
        """Get training configuration section."""
        if self._training_config is None:
            self._training_config = TrainingConfigSection.from_dict(self.raw_config)
            self._training_config.validate()
        return self._training_config
    
    @property
    def lora(self) -> LoRAConfigSection:
        """Get LoRA configuration section."""
        if self._lora_config is None:
            self._lora_config = LoRAConfigSection.from_dict(self.raw_config)
            self._lora_config.validate()
        return self._lora_config
    
    @property
    def output(self) -> OutputConfigSection:
        """Get output configuration section."""
        if self._output_config is None:
            self._output_config = OutputConfigSection.from_dict(self.raw_config)
            self._output_config.validate()
        return self._output_config
    
    @property
    def logging(self) -> LoggingConfigSection:
        if self._logging_config is None:
            self._logging_config = LoggingConfigSection.from_dict(self.raw_config)
            self._logging_config.validate()
        return self._logging_config
    
    @property
    def comet(self) -> CometConfigSection:
        """Get Comet ML configuration section."""
        if self._comet_config is None:
            self._comet_config = CometConfigSection.from_dict(self.raw_config)
            self._comet_config.validate()
        return self._comet_config
    
    @property
    def hub(self) -> HubConfigSection:
        if self._hub_config is None:
            self._hub_config = HubConfigSection.from_dict(self.raw_config)
            self._hub_config.validate()
        return self._hub_config
    
    @property 
    def evaluation(self) -> EvaluationConfigSection:
        if self._evaluation_config is None: 
            self._evaluation_config = EvaluationConfigSection.from_dict(self.raw_config)
            self._evaluation_config.validate()
        return self._evaluation_config
    
    @property 
    def generation(self) -> GenerationConfigSection:
        if self._generation_config is None: 
            self._generation_config = GenerationConfigSection.from_dict(self.raw_config)
            self._generation_config.validate()
        return self._generation_config
    
    @property
    def system(self) -> SystemConfigSection:
        """Get system configuration section."""
        if self._system_config is None:
            self._system_config = SystemConfigSection.from_dict(self.raw_config)
            self._system_config.validate()
        return self._system_config
    
    def validate_all(self) -> None:
        """Validate all configuration sections."""
        sections = [
            self.model, self.dataset, self.training, 
            self.lora, self.output, self.comet, self.system, self.logging, self.hub, self.evaluation
        ]
        
        errors = []
        for section in sections:
            try:
                section.validate()
            except ValidationError as e:
                errors.append(e)
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed with {len(errors)} errors", errors)
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.training.per_device_train_batch_size * self.training.gradient_accumulation_steps
    
    def get_output_dir(self) -> str:
        """Get full output directory path."""
        return self.output.get_output_dir()
    
    def summary(self) -> str:
        """Get configuration summary."""
        return f"""
Configuration Summary:
======================
Model: {self.model.name}
Dataset: {self.dataset.name}
Max Steps: {self.training.max_steps:,}
Batch Size: {self.training.per_device_train_batch_size}
Learning Rate: {self.training.learning_rate:.2e}
LoRA Rank: {self.lora.r}
Output: {self.output.get_output_dir()}
Effective Batch Size: {self.get_effective_batch_size()}
"""


def create_config_manager(
    config_path: Optional[str] = None,
    config_data: Optional[Dict[str, Any]] = None,
    apply_env: bool = True,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> ConfigManager:
    """
    Factory function to create ConfigManager with all overrides applied.
    
    Args:
        config_path: Path to YAML config file
        config_data: Direct config data dictionary
        apply_env: Whether to apply environment variable overrides
        cli_overrides: CLI argument overrides
        
    Returns:
        Configured ConfigManager instance
    """
    manager = ConfigManager(config_path=config_path, config_data=config_data)
    
    if apply_env:
        manager.apply_env_overrides()
    
    if cli_overrides:
        manager.apply_overrides(cli_overrides)
    
    # Validate all sections
    manager.validate_all()
    
    return manager
