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
                
                # Debug SFTTrainer parameters
                logger.info(f"🔧 SFT params debug:")
                logger.info(f"   max_seq_length={config.model.max_seq_length}")
                logger.info(f"   dataset_num_proc={config.dataset.num_proc}")
                logger.info(f"   packing={config.dataset.packing}")
                logger.info(f"   dataset_text_field={config.dataset.text_field}")
                
                # Debug dataset types and lengths
                logger.info(f"🔧 Dataset debug:")
                logger.info(f"   train_dataset type: {type(datasets['train'])}")
                logger.info(f"   train_dataset length: {len(datasets['train'])}")
                if "eval" in datasets and datasets["eval"] is not None:
                    logger.info(f"   eval_dataset type: {type(datasets['eval'])}")
                    logger.info(f"   eval_dataset length: {len(datasets['eval'])}")
                
                # Debug training args object
                logger.info(f"🔧 TrainingArgs debug:")
                logger.info(f"   training_args type: {type(training_args)}")
                logger.info(f"   output_dir: {training_args.output_dir}")
                logger.info(f"   max_steps: {training_args.max_steps}")
                logger.info(f"   per_device_train_batch_size: {training_args.per_device_train_batch_size}")
                
                # Prepare SFTTrainer arguments 
                sft_args = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "train_dataset": datasets["train"],
                    "args": training_args,
                    "data_collator": data_collator,
                    "dataset_text_field": config.dataset.text_field,
                    "max_seq_length": config.model.max_seq_length,
                    "dataset_num_proc": config.dataset.num_proc or 2,  # Safe default
                    "packing": config.dataset.packing if config.dataset.packing is not None else False,  # Safe default
                }
                
                # Only add eval_dataset if it exists
                eval_dataset = datasets.get("eval")
                if eval_dataset is not None and len(eval_dataset) > 0:
                    sft_args["eval_dataset"] = eval_dataset
                    logger.info(f"   eval_dataset: {len(eval_dataset)} samples")
                else:
                    logger.info("   eval_dataset: None (skipping evaluation)")
                
                # Apply train_on_responses_only if configured
                try:
                    logger.info("🔧 Creating SFTTrainer with arguments...")
                    for key, value in sft_args.items():
                        if key != "train_dataset" and key != "eval_dataset":  # Skip large objects
                            logger.info(f"   {key}={value}")
                    
                    trainer = SFTTrainer(**sft_args)
                    logger.info("✅ SFTTrainer created successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create SFTTrainer: {e}")
                    logger.error(f"❌ Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"❌ SFTTrainer traceback: {traceback.format_exc()}")
                    raise
                
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
                
                # Prepare SFTTrainer arguments
                sft_args = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "train_dataset": datasets["train"],
                    "args": training_args,
                    "data_collator": data_collator,
                    "dataset_text_field": config.dataset.text_field,
                    "max_seq_length": config.model.max_seq_length,
                    "dataset_num_proc": config.dataset.num_proc or 2,
                    "packing": config.dataset.packing if config.dataset.packing is not None else False,
                }
                
                # Only add eval_dataset if it exists
                eval_dataset = datasets.get("eval")
                if eval_dataset is not None and len(eval_dataset) > 0:
                    sft_args["eval_dataset"] = eval_dataset
                
                try:
                    logger.info("🔧 Creating standard SFTTrainer with arguments...")
                    for key, value in sft_args.items():
                        if key != "train_dataset" and key != "eval_dataset":  # Skip large objects
                            logger.info(f"   {key}={value}")
                    
                    trainer = SFTTrainer(**sft_args)
                    logger.info("✅ Standard SFTTrainer created successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create standard SFTTrainer: {e}")
                    logger.error(f"❌ Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"❌ Standard SFTTrainer traceback: {traceback.format_exc()}")
                    raise
            
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
        
        logger.info(f"🔧 Mixed precision config: fp16={fp16}, bf16={bf16}")
        
        # Debug config values
        logger.info(f"🔧 Training config debug: max_steps={config.training.max_steps}, num_epochs={config.training.num_train_epochs}")
        logger.info(f"🔧 Eval config debug: eval_steps={config.evaluation.eval_steps}, strategy={config.evaluation.strategy}")
        logger.info(f"🔧 Output config debug: save_steps={config.output.save_steps}, save_strategy={config.output.save_strategy}")
        logger.info(f"🔧 Logging config debug: logging_steps={config.logging.steps}, report_to={config.logging.report_to}")
        logger.info(f"🔧 System config debug: seed={config.system.seed}, num_workers={config.system.dataloader_num_workers}")
        
        # Force fp16 for Tesla T4 and pre-Ampere GPUs regardless of config
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            device_capability = torch.cuda.get_device_capability()
            
            # Tesla T4 and pre-Ampere GPUs (compute capability < 8.0) don't support bf16
            if "T4" in device_name or device_capability[0] < 8:
                logger.info(f"🔧 Detected {device_name} (compute {device_capability}) - forcing fp16 for compatibility")
                fp16 = True
                bf16 = False
            elif not fp16 and not bf16:
                # Auto-detect for Ampere+ GPUs only if not explicitly set
                bf16_supported = torch.cuda.is_bf16_supported()
                bf16 = bf16_supported
                fp16 = not bf16_supported
                logger.info(f"🔧 Auto-detected mixed precision: fp16={fp16}, bf16={bf16}")
        else:
            # CPU fallback
            if not fp16 and not bf16:
                fp16 = True
                bf16 = False
        
        # Create base arguments
        logger.info("🔧 Creating TrainingArguments...")
        
        # Prepare arguments with safe defaults
        max_steps_val = config.training.max_steps if config.training.max_steps and config.training.max_steps > 0 else -1
        num_epochs_val = config.training.num_train_epochs if config.training.num_train_epochs and config.training.num_train_epochs > 0 else None
        
        logger.info(f"🔧 Final TrainingArguments values:")
        logger.info(f"   max_steps={max_steps_val}, num_train_epochs={num_epochs_val}")
        
        # Prepare TrainingArguments kwargs
        training_args_kwargs = {
            "output_dir": config.get_output_dir(),
            "max_steps": max_steps_val,
        }
        
        # Only add num_train_epochs if it's a valid positive number
        if num_epochs_val is not None and num_epochs_val > 0:
            training_args_kwargs["num_train_epochs"] = num_epochs_val
            logger.info(f"   ✅ Using num_train_epochs={num_epochs_val}")
        else:
            logger.info(f"   ✅ Skipping num_train_epochs (will use Transformers default)")
        
        try:
            args = TrainingArguments(
                **training_args_kwargs,
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
                save_steps=config.output.save_steps if config.output.save_steps and config.output.save_steps > 0 else 50,
                save_total_limit=config.output.save_total_limit,
                load_best_model_at_end=config.output.load_best_model_at_end,
                
                # Evaluation
                eval_strategy=config.evaluation.strategy,
                eval_steps=config.evaluation.eval_steps if config.evaluation.eval_steps and config.evaluation.eval_steps > 0 else 50,
                eval_accumulation_steps=config.evaluation.eval_accumulation_steps if config.evaluation.eval_accumulation_steps and config.evaluation.eval_accumulation_steps > 0 else None,
                metric_for_best_model=config.evaluation.metric_for_best_model,
                greater_is_better=config.evaluation.greater_is_better,
                
                # Logging
                logging_steps=config.logging.steps if config.logging.steps and config.logging.steps > 0 else 10,
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
            
            logger.info("✅ TrainingArguments created successfully")
            return args
            
        except Exception as e:
            logger.error(f"❌ Failed to create TrainingArguments: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
    
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
