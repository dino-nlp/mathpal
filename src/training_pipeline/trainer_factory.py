"""SFTTrainer factory and configuration."""

from typing import Any, Optional, Dict
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from training_pipeline.training_config import TrainingConfig


class TrainerFactory:
    """Factory for creating and configuring SFTTrainer instances."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize TrainerFactory with configuration."""
        self.config = config
    
    def create_sft_config(
        self,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> SFTConfig:
        """
        Create SFTConfig from training configuration.
        
        Args:
            config_overrides: Optional configuration overrides
            
        Returns:
            Configured SFTConfig instance
        """
        # Get base configuration
        sft_kwargs = self.config.to_sft_config_kwargs()
        
        # Apply overrides if provided
        if config_overrides:
            sft_kwargs.update(config_overrides)
        
        return SFTConfig(**sft_kwargs)
    
    def create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        sft_config_overrides: Optional[Dict[str, Any]] = None
    ) -> SFTTrainer:
        """
        Create optimized SFTTrainer instance.
        
        Args:
            model: Prepared model for training
            tokenizer: Model tokenizer/processor
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            sft_config_overrides: Optional SFTConfig overrides
            
        Returns:
            Configured SFTTrainer instance
        """
        print("Creating SFTTrainer...")
        
        # Create training arguments
        training_args = self.create_sft_config(sft_config_overrides)
        
        # Initialize trainer
        trainer_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": train_dataset,
            "args": training_args,
        }
        
        # Add eval dataset if provided
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        
        trainer = SFTTrainer(**trainer_kwargs)
        
        # Configure for response-only training if enabled
        if self.config.train_on_responses_only:
            trainer = self._setup_response_only_training(trainer)
        
        print("SFTTrainer created successfully!")
        return trainer
    
    def _setup_response_only_training(self, trainer: SFTTrainer) -> SFTTrainer:
        """
        Configure trainer for response-only training using Unsloth.
        
        Args:
            trainer: SFTTrainer instance
            
        Returns:
            Configured trainer for response-only training
        """
        print("Configuring response-only training...")
        
        try:
            # Use Unsloth's train_on_responses_only for Gemma3N
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
            print("Response-only training configured!")
            return trainer
            
        except Exception as e:
            print(f"Warning: Could not configure response-only training: {e}")
            print("Falling back to standard training...")
            return trainer
    
    def create_training_pipeline(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        sft_config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create complete training pipeline.
        
        Args:
            model: Model for training
            tokenizer: Model tokenizer/processor
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            sft_config_overrides: Optional SFTConfig overrides
            
        Returns:
            Dictionary containing trainer and configuration info
        """
        # Create trainer
        trainer = self.create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            sft_config_overrides=sft_config_overrides
        )
        
        # Prepare model for training (if using Unsloth)
        try:
            from unsloth import FastModel
            FastModel.for_training(model)
            print("Model prepared for training with Unsloth optimizations!")
        except:
            print("Model prepared for standard training")
        
        return {
            "trainer": trainer,
            "config": self.config,
            "training_args": trainer.args,
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset
        }
    
    def print_training_info(
        self, 
        trainer: SFTTrainer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Print training configuration information.
        
        Args:
            trainer: SFTTrainer instance
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        args = trainer.args
        
        print("\n🚀 Training Configuration:")
        print(f"   Output directory: {args.output_dir}")
        print(f"   Max steps: {args.max_steps}")
        print(f"   Per device batch size: {args.per_device_train_batch_size}")
        print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"   Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Warmup ratio: {args.warmup_ratio}")
        print(f"   Weight decay: {args.weight_decay}")
        print(f"   Optimizer: {args.optim}")
        print(f"   LR scheduler: {args.lr_scheduler_type}")
        print(f"   Logging steps: {args.logging_steps}")
        print(f"   Save steps: {args.save_steps}")
        print(f"   Report to: {args.report_to}")
        print(f"   Seed: {args.seed}")
        
        print(f"\n📊 Dataset Information:")
        print(f"   Training samples: {len(train_dataset):,}")
        if eval_dataset:
            print(f"   Evaluation samples: {len(eval_dataset):,}")
        
        # Calculate training estimates
        total_samples = len(train_dataset)
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        steps_per_epoch = total_samples // batch_size
        
        print(f"\n⏱️ Training Estimates:")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total epochs: ~{args.max_steps / steps_per_epoch:.2f}")
        print(f"   Samples per step: {batch_size}")
    
    @staticmethod
    def get_recommended_configs() -> Dict[str, Dict[str, Any]]:
        """
        Get recommended training configurations for different scenarios.
        
        Returns:
            Dictionary of configuration presets
        """
        return {
            "quick_test": {
                "max_steps": 10,
                "logging_steps": 1,
                "save_steps": 5,
                "description": "Quick test configuration"
            },
            "development": {
                "max_steps": 100,
                "logging_steps": 5,
                "save_steps": 25,
                "description": "Development and debugging"
            },
            "production": {
                "max_steps": 1000,
                "logging_steps": 10,
                "save_steps": 100,
                "description": "Production training"
            },
            "memory_efficient": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "use_gradient_checkpointing": True,
                "description": "Memory efficient for limited resources"
            }
        }