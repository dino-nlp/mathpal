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

from ..core.exceptions import ConfigurationError, ValidationError

# Type variable for config sections
T = TypeVar('T', bound='ConfigSection')


# =====================================================================================
# LEGACY BACKWARD COMPATIBILITY CLASSES
# =====================================================================================
# These classes are maintained for backward compatibility with existing code
# that still imports from enhanced_config.py

@dataclass
class ModelConfig:
    """Legacy model configuration for backward compatibility."""
    name: str = "unsloth/gemma-3n-E4B-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False


@dataclass 
class DatasetConfig:
    """Legacy dataset configuration for backward compatibility."""
    name: str = "ngohongthai/exam-sixth_grade-instruct-dataset"
    train_split: str = "train"
    test_split: str = "test"
    text_field: str = "text"
    max_length: Optional[int] = None
    num_proc: int = 2
    packing: bool = True


@dataclass
class TrainingConfig:
    """Legacy training configuration for backward compatibility."""
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


@dataclass
class LoRAConfig:
    """Legacy LoRA configuration for backward compatibility."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_rslora: bool = False
    loftq_config: Optional[Dict[str, Any]] = None


@dataclass
class SystemConfig:
    """Legacy system configuration for backward compatibility."""
    seed: int = 42
    use_gradient_checkpointing: Union[str, bool] = "unsloth"
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0


@dataclass
class OutputConfig:
    """Legacy output configuration for backward compatibility."""
    base_dir: str = "outputs"
    experiment_name: str = "gemma3n-vietnamese-math"
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    save_formats: List[str] = field(default_factory=lambda: ["lora", "merged_16bit"])


@dataclass
class EvaluationConfig:
    """Legacy evaluation configuration for backward compatibility."""
    strategy: str = "steps"
    eval_steps: int = 50
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16_full_eval: bool = True
    eval_accumulation_steps: int = 4


@dataclass
class LoggingConfig:
    """Legacy logging configuration for backward compatibility."""
    steps: int = 10
    report_to: str = "none"
    level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class CometConfig:
    """Legacy Comet ML configuration for backward compatibility."""
    enabled: bool = False
    experiment_name: str = "gemma3n-math-tutor"
    tags: List[str] = field(default_factory=lambda: [
        "gemma3n", "vietnamese", "math-education", "6th-grade", "instruction-tuning"
    ])
    auto_metric_logging: bool = True
    auto_param_logging: bool = True
    auto_histogram_weight_logging: bool = False
    auto_histogram_gradient_logging: bool = False
    auto_histogram_activation_logging: bool = False
    auto_output_logging: str = "default"
    log_model: bool = True
    log_graph: bool = False
    log_code: bool = True
    log_git_metadata: bool = True


@dataclass
class InferenceConfig:
    """Legacy inference configuration for backward compatibility."""
    test_after_training: bool = True
    num_test_examples: int = 5
    generation: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": None
    })


@dataclass
class HubConfig:
    """Legacy Hub configuration for backward compatibility."""
    push_to_hub: bool = False
    username: Optional[str] = None
    repo_name: Optional[str] = None
    private: bool = True
    token: Optional[str] = None


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
    max_length: Optional[int] = None
    num_proc: int = 2
    packing: bool = False
    
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
                'max_length': data.get('max_length', cls.max_length),
                'num_proc': data.get('num_proc', cls.num_proc),
                'packing': data.get('packing', cls.packing),
            }
        
        return cls(
            name=dataset_data.get('name', cls.name),
            train_split=dataset_data.get('train_split', cls.train_split),
            test_split=dataset_data.get('test_split', cls.test_split),
            text_field=dataset_data.get('text_field', cls.text_field),
            max_length=dataset_data.get('max_length', cls.max_length),
            num_proc=dataset_data.get('num_proc', cls.num_proc),
            packing=dataset_data.get('packing', cls.packing),
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
class CometConfigSection(ConfigSection):
    """Comet ML configuration section."""
    enabled: bool = False
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
                'enabled': data.get('report_to') == 'comet_ml',
                'experiment_name': data.get('experiment_name', cls.experiment_name),
                'tags': data.get('tags', default_tags),
                'auto_metric_logging': data.get('auto_metric_logging', cls.auto_metric_logging),
                'auto_param_logging': data.get('auto_param_logging', cls.auto_param_logging),
            }
        
        default_tags = ["gemma3n", "vietnamese", "math"]
        return cls(
            enabled=comet_data.get('enabled', cls.enabled),
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
    def comet(self) -> CometConfigSection:
        """Get Comet ML configuration section."""
        if self._comet_config is None:
            self._comet_config = CometConfigSection.from_dict(self.raw_config)
            self._comet_config.validate()
        return self._comet_config
    
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
            self.lora, self.output, self.comet, self.system
        ]
        
        errors = []
        for section in sections:
            try:
                section.validate()
            except ValidationError as e:
                errors.append(e)
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed with {len(errors)} errors", errors)
    
    def to_comprehensive_config(self):
        """Convert to ComprehensiveTrainingConfig for backward compatibility."""
        # Use local classes instead of importing from enhanced_config
        
        return ComprehensiveTrainingConfig(
            model=ModelConfig(
                name=self.model.name,
                max_seq_length=self.model.max_seq_length,
                load_in_4bit=self.model.load_in_4bit,
                load_in_8bit=self.model.load_in_8bit,
                full_finetuning=self.model.full_finetuning,
            ),
            dataset=DatasetConfig(
                name=self.dataset.name,
                train_split=self.dataset.train_split,
                test_split=self.dataset.test_split,
                text_field=self.dataset.text_field,
                max_length=self.dataset.max_length,
                num_proc=self.dataset.num_proc,
                packing=self.dataset.packing,
            ),
            training=TrainingConfig(
                max_steps=self.training.max_steps,
                num_train_epochs=self.training.num_train_epochs,
                per_device_train_batch_size=self.training.per_device_train_batch_size,
                per_device_eval_batch_size=self.training.per_device_eval_batch_size,
                gradient_accumulation_steps=self.training.gradient_accumulation_steps,
                learning_rate=self.training.learning_rate,
                warmup_ratio=self.training.warmup_ratio,
                weight_decay=self.training.weight_decay,
                optim=self.training.optim,
                lr_scheduler_type=self.training.lr_scheduler_type,
                max_grad_norm=self.training.max_grad_norm,
                fp16=self.training.fp16,
                bf16=self.training.bf16,
                train_on_responses_only=self.training.train_on_responses_only,
            ),
            lora=LoRAConfig(
                r=self.lora.r,
                alpha=self.lora.alpha,
                dropout=self.lora.dropout,
                bias=self.lora.bias,
                target_modules=self.lora.target_modules,
                use_rslora=self.lora.use_rslora,
            ),
            output=OutputConfig(
                base_dir=self.output.base_dir,
                experiment_name=self.output.experiment_name,
                save_strategy=self.output.save_strategy,
                save_steps=self.output.save_steps,
                save_total_limit=self.output.save_total_limit,
                load_best_model_at_end=self.output.load_best_model_at_end,
                save_formats=self.output.save_formats,
            ),
            system=SystemConfig(
                seed=self.system.seed,
                use_gradient_checkpointing=self.system.use_gradient_checkpointing,
                dataloader_drop_last=self.system.dataloader_drop_last,
                dataloader_pin_memory=self.system.dataloader_pin_memory,
                dataloader_num_workers=self.system.dataloader_num_workers,
            ),
            comet=CometConfig(
                enabled=self.comet.enabled,
                experiment_name=self.comet.experiment_name,
                tags=self.comet.tags,
                auto_metric_logging=self.comet.auto_metric_logging,
                auto_param_logging=self.comet.auto_param_logging,
            ),
        )
    
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


# =====================================================================================
# LEGACY COMPREHENSIVE TRAINING CONFIG (BACKWARD COMPATIBILITY)
# =====================================================================================

@dataclass
class ComprehensiveTrainingConfig:
    """Legacy comprehensive training configuration for backward compatibility."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    comet: CometConfig = field(default_factory=CometConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        errors = []
        
        # Model validation
        if not self.model.name:
            errors.append(ValidationError("Model name cannot be empty", "model.name"))
        
        if self.model.max_seq_length <= 0:
            errors.append(ValidationError("max_seq_length must be positive", "model.max_seq_length"))
        
        if self.model.load_in_4bit and self.model.load_in_8bit:
            errors.append(ValidationError("Cannot use both 4-bit and 8-bit quantization", "model"))
        
        # Training validation
        if self.training.max_steps <= 0 and (self.training.num_train_epochs is None or self.training.num_train_epochs <= 0):
            errors.append(ValidationError("Must specify either max_steps > 0 or num_train_epochs > 0", "training"))
        
        if self.training.per_device_train_batch_size <= 0:
            errors.append(ValidationError("per_device_train_batch_size must be positive", "training.per_device_train_batch_size"))
        
        if self.training.gradient_accumulation_steps <= 0:
            errors.append(ValidationError("gradient_accumulation_steps must be positive", "training.gradient_accumulation_steps"))
        
        if self.training.learning_rate <= 0:
            errors.append(ValidationError("learning_rate must be positive", "training.learning_rate"))
        
        # LoRA validation
        if self.lora.r <= 0:
            errors.append(ValidationError("LoRA rank must be positive", "lora.r"))
        
        if self.lora.alpha <= 0:
            errors.append(ValidationError("LoRA alpha must be positive", "lora.alpha"))
        
        if not (0 <= self.lora.dropout <= 1):
            errors.append(ValidationError("LoRA dropout must be between 0 and 1", "lora.dropout"))
        
        # Output validation
        if not self.output.base_dir:
            errors.append(ValidationError("Output base directory cannot be empty", "output.base_dir"))
        
        if not self.output.experiment_name:
            errors.append(ValidationError("Experiment name cannot be empty", "output.experiment_name"))
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed with {len(errors)} errors", errors)
    
    def get_output_dir(self) -> str:
        """Get full output directory path."""
        return f"{self.output.base_dir}/{self.output.experiment_name}"
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.training.per_device_train_batch_size * self.training.gradient_accumulation_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Convert dataclass to dict
                result[field_name] = {k: v for k, v in field_value.__dict__.items()}
            else:
                result[field_name] = field_value
        return result
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


class ConfigLoader:
    """Legacy config loader for backward compatibility."""
    
    @staticmethod
    def load_from_yaml(yaml_path: str) -> ComprehensiveTrainingConfig:
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        try:
            return ConfigLoader._dict_to_config(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config file {yaml_path}: {e}")
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ComprehensiveTrainingConfig:
        """Convert dictionary to config object."""
        
        # Extract nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        lora_config = LoRAConfig(**config_dict.get('lora', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        comet_config = CometConfig(**config_dict.get('comet', {}))
        inference_config = InferenceConfig(**config_dict.get('inference', {}))
        hub_config = HubConfig(**config_dict.get('hub', {}))
        
        return ComprehensiveTrainingConfig(
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            lora=lora_config,
            system=system_config,
            output=output_config,
            evaluation=evaluation_config,
            logging=logging_config,
            comet=comet_config,
            inference=inference_config,
            hub=hub_config
        )
    
    @staticmethod
    def apply_quick_test_profile(config: ComprehensiveTrainingConfig) -> ComprehensiveTrainingConfig:
        """Apply quick test profile overrides."""
        config.training.max_steps = 20
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 4
        config.model.max_seq_length = 1024
        config.output.experiment_name = "quick-test"
        config.output.save_steps = 10
        config.logging.steps = 2
        config.evaluation.strategy = "steps"
        config.evaluation.eval_steps = 10
        return config
    
    @staticmethod
    def apply_cli_overrides(config: ComprehensiveTrainingConfig, 
                          experiment_name: Optional[str] = None,
                          output_dir: Optional[str] = None,
                          max_steps: Optional[int] = None) -> ComprehensiveTrainingConfig:
        """Apply CLI argument overrides."""
        if experiment_name:
            config.output.experiment_name = experiment_name
        if output_dir:
            config.output.base_dir = output_dir
        if max_steps:
            config.training.max_steps = max_steps
        return config
