"""Factory for creating models and tokenizers."""

from typing import Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from training_pipeline.utils.exceptions import UnsupportedModelError, ModelError
from training_pipeline.config.config_manager import ConfigManager
from training_pipeline.utils import get_logger
from unsloth import FastModel, get_chat_template

logger = get_logger()


class ModelFactory:
    """Factory for creating models and tokenizers based on configuration."""
    
    @staticmethod
    def create_model(config: ConfigManager) -> Tuple[Any, Any]:
        """
        Create model and tokenizer based on configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            UnsupportedModelError: If model type is not supported
            ModelError: If model creation fails
        """
        logger.info(f"🚀 Creating Unsloth model: {config.model.name}")
            
        # Create model with Unsloth optimizations
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.model.name,
            dtype=None,  # Auto-detect
            max_seq_length=config.model.max_seq_length,
            load_in_4bit=config.model.load_in_4bit,
            load_in_8bit=config.model.load_in_8bit,
            full_finetuning = config.model.full_finetuning,
            use_gradient_checkpointing=config.system.use_gradient_checkpointing
            # token = "hf_...", # use one if using gated models
        )
        
        # Apply LoRA if not doing full fine-tuning
        if not config.model.full_finetuning:
            logger.info("🔧 Applying LoRA configuration...")
            
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # SHould leave on always!
                r=config.lora.r,
                target_modules=config.lora.target_modules,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                bias=config.lora.bias,
                random_state=config.system.seed
            )
        
        # Setup chat template for Gemma3N
        tokenizer = get_chat_template(tokenizer, "gemma-3n")
        logger.info("📊 Model loaded successfully")
        return model, tokenizer
       
    @staticmethod
    def estimate_memory_usage(config: ConfigManager) -> float:
        """
        Estimate memory usage for model configuration.
        
        Returns:
            Estimated memory usage in GB
        """
        # Basic estimation based on model size and quantization
        base_memory = 8.0  # Base memory for Gemma-3n-4B
        
        # Adjust for quantization
        if config.model.load_in_4bit:
            base_memory *= 0.25
        elif config.model.load_in_8bit:
            base_memory *= 0.5
        
        # Adjust for batch size
        memory_per_sample = 0.1  # GB per sample
        total_batch_size = (config.training.per_device_train_batch_size * 
                          config.training.gradient_accumulation_steps)
        batch_memory = total_batch_size * memory_per_sample
        
        # Add LoRA overhead (minimal)
        if not config.model.full_finetuning:
            lora_memory = 0.1 * (config.lora.r / 16)  # Scale with rank
        else:
            lora_memory = base_memory * 2  # Full fine-tuning doubles memory
        
        total_memory = base_memory + batch_memory + lora_memory
        
        logger.debug(f"💾 Estimated memory usage: {total_memory:.2f} GB")
        return total_memory
