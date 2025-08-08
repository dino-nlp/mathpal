"""Training configuration for Gemma3N fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from base_config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for Gemma3N training pipeline."""
    
    # Model settings
    model_name: str = "unsloth/gemma-3n-E2B-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    full_finetuning: bool = False
    
    # Dataset settings  
    dataset_name: str = "ngohongthai/exam-sixth_grade-instruct-dataset"
    train_split: str = "train"
    test_split: str = "test"
    
    # Output settings
    output_dir: str = "outputs/gemma3n-finetune"
    experiment_name: str = "baseline"
    
    # Training hyperparameters
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 5
    save_steps: int = 50
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: Optional[list] = None  # Will use Unsloth defaults
    
    # System settings
    use_gradient_checkpointing: bool = False
    report_to: str = "comet_ml"  # Options: "comet_ml", "tensorboard", "wandb", "none"
    seed: int = 42
    
    # Optimizer settings
    optim: str = "adamw_torch_fused"
    lr_scheduler_type: str = "cosine"
    
    # Additional training settings
    train_on_responses_only: bool = True
    dataset_text_field: str = "text"
    max_length: Optional[int] = None  # Will use max_seq_length if None
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if self.report_to not in ["comet_ml", "tensorboard", "wandb", "none"]:
            raise ValueError(f"Invalid report_to value: {self.report_to}")
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    def get_output_dir(self) -> str:
        """Get full output directory path."""
        return f"{self.output_dir}/{self.experiment_name}"
    
    def get_max_length(self) -> int:
        """Get max length for training."""
        return self.max_length or self.max_seq_length
    
    def to_sft_config_kwargs(self) -> Dict[str, Any]:
        """Convert to SFTConfig kwargs."""
        return {
            "dataset_text_field": self.dataset_text_field,
            "output_dir": self.get_output_dir(),
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "logging_steps": self.logging_steps,
            "save_strategy": "steps",
            "save_steps": self.save_steps,
            "report_to": self.report_to,
            "max_length": self.get_max_length(),
            "seed": self.seed,
        }
    
    def to_lora_config_kwargs(self) -> Dict[str, Any]:
        """Convert to LoRA config kwargs."""
        kwargs = {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.lora_bias,
            "random_state": self.seed,
        }
        
        if self.lora_target_modules:
            kwargs["target_modules"] = self.lora_target_modules
            
        return kwargs