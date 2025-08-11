"""Enhanced configuration classes for the new architecture."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
import yaml
from pathlib import Path

from .exceptions import ValidationError, ConfigurationError
from ..config.base_config import BaseConfig


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "unsloth/gemma-3n-E4B-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False


@dataclass 
class DatasetConfig:
    """Dataset configuration."""
    name: str = "ngohongthai/exam-sixth_grade-instruct-dataset"
    train_split: str = "train"
    test_split: str = "test"
    text_field: str = "text"
    max_length: Optional[int] = None
    num_proc: int = 2
    packing: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
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
    """LoRA configuration."""
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
    """System and optimization configuration."""
    seed: int = 42
    use_gradient_checkpointing: str = "unsloth"  # "unsloth", True, False
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0


@dataclass
class OutputConfig:
    """Output and saving configuration."""
    base_dir: str = "outputs"
    experiment_name: str = "gemma3n-vietnamese-math"
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    save_formats: List[str] = field(default_factory=lambda: ["lora", "merged_16bit"])


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    strategy: str = "steps"
    eval_steps: int = 50
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16_full_eval: bool = True
    eval_accumulation_steps: int = 4


@dataclass
class LoggingConfig:
    """Logging configuration."""
    steps: int = 10
    report_to: str = "none"
    level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class CometConfig:
    """Comet ML configuration."""
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
    """Inference and testing configuration."""
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
    """HuggingFace Hub configuration."""
    push_to_hub: bool = False
    username: Optional[str] = None
    repo_name: Optional[str] = None
    private: bool = True
    token: Optional[str] = None


@dataclass
class ComprehensiveTrainingConfig(BaseConfig):
    """Comprehensive training configuration matching YAML structure."""
    
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
    
    def to_legacy_training_config(self):
        """Convert to legacy TrainingConfig for backward compatibility."""
        from ..config.training_config import TrainingConfig as LegacyTrainingConfig
        
        return LegacyTrainingConfig(
            model_name=self.model.name,
            max_seq_length=self.model.max_seq_length,
            load_in_4bit=self.model.load_in_4bit,
            full_finetuning=self.model.full_finetuning,
            dataset_name=self.dataset.name,
            train_split=self.dataset.train_split,
            test_split=self.dataset.test_split,
            output_dir=self.output.base_dir,
            experiment_name=self.output.experiment_name,
            max_steps=self.training.max_steps,
            per_device_train_batch_size=self.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.training.gradient_accumulation_steps,
            learning_rate=self.training.learning_rate,
            warmup_ratio=self.training.warmup_ratio,
            weight_decay=self.training.weight_decay,
            logging_steps=self.logging.steps,
            save_steps=self.output.save_steps,
            lora_r=self.lora.r,
            lora_alpha=self.lora.alpha,
            lora_dropout=self.lora.dropout,
            lora_bias=self.lora.bias,
            lora_target_modules=self.lora.target_modules,
            use_gradient_checkpointing=self.system.use_gradient_checkpointing == "unsloth" or self.system.use_gradient_checkpointing is True,
            report_to=self.logging.report_to,
            seed=self.system.seed,
            optim=self.training.optim,
            lr_scheduler_type=self.training.lr_scheduler_type,
            train_on_responses_only=self.training.train_on_responses_only,
            dataset_text_field=self.dataset.text_field,
            max_length=self.dataset.max_length
        )


class ConfigLoader:
    """Enhanced config loader for new architecture."""
    
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
