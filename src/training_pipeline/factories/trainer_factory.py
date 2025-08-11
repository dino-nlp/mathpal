"""Factory for creating trainers based on configuration."""

from typing import Any, Dict, Optional
import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from ..core.exceptions import TrainingError, UnsupportedModelError
from ..core.enhanced_config import ComprehensiveTrainingConfig
from ..utils import get_logger

logger = get_logger()


class TrainerFactory:
    """Factory for creating trainers based on configuration."""
    
    SUPPORTED_TRAINER_TYPES = ["sft", "dpo", "rlhf"]
    
    @staticmethod
    def create_trainer(config: ComprehensiveTrainingConfig,
                      model: Any,
                      tokenizer: Any,
                      datasets: Dict[str, Any]) -> Any:
        """
        Create trainer based on configuration.
        
        Args:
            config: Training configuration
            model: Model to train
            tokenizer: Tokenizer
            datasets: Dictionary containing train and optional eval datasets
            
        Returns:
            Configured trainer
            
        Raises:
            TrainingError: If trainer creation fails
        """
        try:
            # Default to SFT (Supervised Fine-Tuning)
            trainer_type = getattr(config.training, 'method', 'sft')
            
            if trainer_type == 'sft':
                return TrainerFactory._create_sft_trainer(config, model, tokenizer, datasets)
            elif trainer_type == 'dpo':
                return TrainerFactory._create_dpo_trainer(config, model, tokenizer, datasets)
            else:
                raise TrainingError(f"Unsupported trainer type: {trainer_type}")
                
        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            raise TrainingError(f"Failed to create trainer: {e}")
    
    @staticmethod
    def _create_sft_trainer(config: ComprehensiveTrainingConfig,
                           model: Any,
                           tokenizer: Any,
                           datasets: Dict[str, Any]) -> Any:
        """Create SFT (Supervised Fine-Tuning) trainer."""
        try:
            from trl import SFTTrainer
            
            logger.info("🏋️ Creating SFT Trainer...")
            
            # Create training arguments
            training_args = TrainerFactory._create_training_arguments(config)
            
            # Create data collator
            data_collator = TrainerFactory._create_data_collator(config, tokenizer)
            
            # Determine if we should use Unsloth optimizations
            use_unsloth = TrainerFactory._should_use_unsloth(config, model)
            
            if use_unsloth:
                logger.info("⚡ Using Unsloth optimizations")
                
                # Apply train_on_responses_only if configured
                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=datasets["train"],
                    eval_dataset=datasets.get("eval"),
                    args=training_args,
                    data_collator=data_collator,
                    dataset_text_field=config.dataset.text_field,
                    max_seq_length=config.model.max_seq_length,
                    dataset_num_proc=config.dataset.num_proc,
                    packing=config.dataset.packing,
                )
                
                # Apply Unsloth's train_on_responses_only optimization
                if config.training.train_on_responses_only:
                    try:
                        from unsloth.chat_templates import train_on_responses_only
                        trainer = train_on_responses_only(trainer)
                        logger.info("✅ Applied train_on_responses_only optimization")
                    except ImportError:
                        logger.warning("⚠️ Could not import train_on_responses_only from Unsloth")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to apply train_on_responses_only: {e}")
            else:
                logger.info("🤗 Using standard HuggingFace trainer")
                
                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=datasets["train"],
                    eval_dataset=datasets.get("eval"),
                    args=training_args,
                    data_collator=data_collator,
                    dataset_text_field=config.dataset.text_field,
                    max_seq_length=config.model.max_seq_length,
                    dataset_num_proc=config.dataset.num_proc,
                    packing=config.dataset.packing,
                )
            
            # Print training info
            TrainerFactory._print_training_info(trainer, datasets, config)
            
            return trainer
            
        except ImportError:
            raise TrainingError(
                "TRL not installed. Please install with: pip install trl"
            )
        except Exception as e:
            raise TrainingError(f"Failed to create SFT trainer: {e}")
    
    @staticmethod
    def _create_dpo_trainer(config: ComprehensiveTrainingConfig,
                           model: Any,
                           tokenizer: Any,
                           datasets: Dict[str, Any]) -> Any:
        """Create DPO (Direct Preference Optimization) trainer."""
        try:
            from trl import DPOTrainer, DPOConfig
            
            logger.info("🎯 Creating DPO Trainer...")
            
            # Create DPO-specific config
            dpo_config = DPOConfig(
                output_dir=config.get_output_dir(),
                per_device_train_batch_size=config.training.per_device_train_batch_size,
                per_device_eval_batch_size=config.training.per_device_eval_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                learning_rate=config.training.learning_rate,
                max_steps=config.training.max_steps,
                warmup_ratio=config.training.warmup_ratio,
                logging_steps=config.logging.steps,
                save_strategy=config.output.save_strategy,
                save_steps=config.output.save_steps,
                evaluation_strategy=config.evaluation.strategy,
                eval_steps=config.evaluation.eval_steps,
                bf16=config.training.bf16,
                fp16=config.training.fp16,
                optim=config.training.optim,
                lr_scheduler_type=config.training.lr_scheduler_type,
                report_to=config.logging.report_to,
                seed=config.system.seed,
                beta=0.1,  # DPO beta parameter
                max_prompt_length=512,
                max_length=config.model.max_seq_length,
            )
            
            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=datasets["train"],
                eval_dataset=datasets.get("eval"),
                tokenizer=tokenizer,
            )
            
            return trainer
            
        except ImportError:
            raise TrainingError(
                "TRL not installed. Please install with: pip install trl"
            )
        except Exception as e:
            raise TrainingError(f"Failed to create DPO trainer: {e}")
    
    @staticmethod
    def _create_training_arguments(config: ComprehensiveTrainingConfig) -> TrainingArguments:
        """Create TrainingArguments from configuration."""
                # Handle mixed precision - Tesla T4 compatibility
        fp16 = config.training.fp16
        bf16 = config.training.bf16
        
        # Auto-detect if not explicitly set, with Tesla T4 compatibility
        if not fp16 and not bf16:
            # Check for bf16 support more thoroughly 
            bf16_supported = torch.cuda.is_bf16_supported()
            
            # Additional check for Tesla T4 and other pre-Ampere GPUs
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name()
                device_capability = torch.cuda.get_device_capability()
                
                # Tesla T4 and pre-Ampere GPUs (compute capability < 8.0) don't support bf16
                if "T4" in device_name or device_capability[0] < 8:
                    bf16_supported = False
                    logger.info(f"🔧 Detected {device_name} (compute {device_capability}) - using fp16 instead of bf16")
            
            bf16 = bf16_supported
            fp16 = not bf16_supported
        
        # Create base arguments
        args = TrainingArguments(
            output_dir=config.get_output_dir(),
            
            # Training parameters
            max_steps=config.training.max_steps if config.training.max_steps > 0 else -1,
            num_train_epochs=config.training.num_train_epochs if config.training.num_train_epochs else None,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            per_device_eval_batch_size=config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_ratio=config.training.warmup_ratio,
            optim=config.training.optim,
            lr_scheduler_type=config.training.lr_scheduler_type,
            max_grad_norm=config.training.max_grad_norm,
            
            # Mixed precision
            fp16=fp16,
            bf16=bf16,
            fp16_full_eval=config.evaluation.fp16_full_eval,
            
            # Saving
            save_strategy=config.output.save_strategy,
            save_steps=config.output.save_steps,
            save_total_limit=config.output.save_total_limit,
            load_best_model_at_end=config.output.load_best_model_at_end,
            
            # Evaluation
            eval_strategy=config.evaluation.strategy,
            eval_steps=config.evaluation.eval_steps,
            eval_accumulation_steps=config.evaluation.eval_accumulation_steps,
            metric_for_best_model=config.evaluation.metric_for_best_model,
            greater_is_better=config.evaluation.greater_is_better,
            
            # Logging
            logging_steps=config.logging.steps,
            report_to=config.logging.report_to,
            
            # System
            dataloader_drop_last=config.system.dataloader_drop_last,
            dataloader_pin_memory=config.system.dataloader_pin_memory,
            dataloader_num_workers=config.system.dataloader_num_workers,
            seed=config.system.seed,
            
            # Gradient checkpointing
            gradient_checkpointing=bool(config.system.use_gradient_checkpointing),
            
            # Additional optimizations
            remove_unused_columns=False,  # Keep all columns for SFT
            ddp_find_unused_parameters=False,
        )
        
        return args
    
    @staticmethod
    def _create_data_collator(config: ComprehensiveTrainingConfig, tokenizer: Any) -> Any:
        """Create appropriate data collator."""
        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,  # Will be set by trainer
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=True,
        )
    
    @staticmethod
    def _should_use_unsloth(config: ComprehensiveTrainingConfig, model: Any) -> bool:
        """Determine if we should use Unsloth optimizations."""
        # Check if model was created with Unsloth
        model_name = config.model.name
        return model_name.startswith("unsloth/") or hasattr(model, 'get_peft_model')
    
    @staticmethod
    def _print_training_info(trainer: Any, datasets: Dict[str, Any], config: ComprehensiveTrainingConfig):
        """Print training information."""
        try:
            train_dataset = datasets["train"]
            eval_dataset = datasets.get("eval")
            
            logger.info("📋 Training Configuration:")
            logger.info(f"   📊 Training samples: {len(train_dataset):,}")
            if eval_dataset:
                logger.info(f"   📊 Evaluation samples: {len(eval_dataset):,}")
            
            logger.info(f"   🎯 Max steps: {config.training.max_steps:,}")
            logger.info(f"   📦 Batch size per device: {config.training.per_device_train_batch_size}")
            logger.info(f"   🔄 Gradient accumulation: {config.training.gradient_accumulation_steps}")
            logger.info(f"   📈 Effective batch size: {config.get_effective_batch_size()}")
            logger.info(f"   📚 Learning rate: {config.training.learning_rate:.2e}")
            logger.info(f"   🎲 LoRA rank: {config.lora.r}")
            logger.info(f"   💾 Mixed precision: {'bf16' if config.training.bf16 else 'fp16' if config.training.fp16 else 'fp32'}")
            
            # Estimate training time
            estimated_time = TrainerFactory._estimate_training_time(config, len(train_dataset))
            logger.info(f"   ⏱️  Estimated training time: {estimated_time}")
            
        except Exception as e:
            logger.warning(f"Could not print training info: {e}")
    
    @staticmethod
    def _estimate_training_time(config: ComprehensiveTrainingConfig, num_samples: int) -> str:
        """Estimate training time based on configuration."""
        try:
            # Basic estimation (very rough)
            samples_per_step = config.get_effective_batch_size()
            total_steps = config.training.max_steps
            
            if total_steps <= 0:
                total_steps = (num_samples // samples_per_step) * (config.training.num_train_epochs or 1)
            
            # Rough estimate: 0.5-2 seconds per step depending on model size and batch size
            seconds_per_step = 1.0
            if config.model.load_in_4bit:
                seconds_per_step *= 0.7
            if config.lora.r > 16:
                seconds_per_step *= 1.2
            
            total_seconds = total_steps * seconds_per_step
            
            if total_seconds < 60:
                return f"{total_seconds:.0f} seconds"
            elif total_seconds < 3600:
                return f"{total_seconds/60:.1f} minutes"
            else:
                return f"{total_seconds/3600:.1f} hours"
                
        except:
            return "Unknown"
