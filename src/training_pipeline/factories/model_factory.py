"""Factory for creating models and tokenizers."""

from typing import Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from training_pipeline.utils.exceptions import UnsupportedModelError, ModelError
from training_pipeline.config.config_manager import ConfigManager
from training_pipeline.utils import get_logger

logger = get_logger()


class ModelFactory:
    """Factory for creating models and tokenizers based on configuration."""
    
    SUPPORTED_MODEL_TYPES = {
        "unsloth": ["unsloth/"],
        "huggingface": ["google/", "meta-llama/", "microsoft/"],
        "gemma": ["google/gemma", "gemma"]
    }
    
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
        try:
            model_name = config.model.name
            
            if ModelFactory._is_unsloth_model(model_name):
                return ModelFactory._create_unsloth_model(config)
            elif ModelFactory._is_huggingface_model(model_name):
                return ModelFactory._create_huggingface_model(config)
            else:
                raise UnsupportedModelError(f"Model {model_name} is not supported")
                
        except Exception as e:
            if isinstance(e, (UnsupportedModelError, ModelError)):
                raise
            raise ModelError(f"Failed to create model {config.model.name}: {e}")
    
    @staticmethod
    def _is_unsloth_model(model_name: str) -> bool:
        """Check if model is an Unsloth model."""
        return any(model_name.startswith(prefix) for prefix in ModelFactory.SUPPORTED_MODEL_TYPES["unsloth"])
    
    @staticmethod
    def _is_huggingface_model(model_name: str) -> bool:
        """Check if model is a standard HuggingFace model."""
        return any(model_name.startswith(prefix) for prefix in ModelFactory.SUPPORTED_MODEL_TYPES["huggingface"])
    
    @staticmethod
    def _create_unsloth_model(config: ConfigManager) -> Tuple[Any, Any]:
        """Create Unsloth optimized model."""
        try:
            from unsloth import FastLanguageModel
            
            logger.info(f"🚀 Creating Unsloth model: {config.model.name}")
            
            # Create model with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model.name,
                max_seq_length=config.model.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=config.model.load_in_4bit,
                load_in_8bit=config.model.load_in_8bit,
                # token=None,  # Add HF token if needed
            )
            
            # Ensure tokenizer has proper pad_token for tensor creation
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"✅ Set pad_token to eos_token: {tokenizer.eos_token}")
            else:
                logger.info(f"✅ Pad_token already set: {tokenizer.pad_token}")
            
            # Apply LoRA if not doing full fine-tuning
            if not config.model.full_finetuning:
                logger.info("🔧 Applying LoRA configuration...")
                
                # Determine gradient checkpointing setting
                use_gradient_checkpointing = config.system.use_gradient_checkpointing
                if use_gradient_checkpointing == "unsloth":
                    use_gradient_checkpointing = "unsloth"
                elif use_gradient_checkpointing in [True, "true", "True"]:
                    use_gradient_checkpointing = True
                else:
                    use_gradient_checkpointing = False
                
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=config.lora.r,
                    target_modules=config.lora.target_modules,
                    lora_alpha=config.lora.alpha,
                    lora_dropout=config.lora.dropout,
                    bias=config.lora.bias,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    random_state=config.system.seed,
                    use_rslora=config.lora.use_rslora,
                    max_seq_length=config.model.max_seq_length,
                )
            
            # Print model info
            try:
                from ..models.model_loader import ModelLoader
                ModelLoader._print_model_info_static(model)
            except:
                logger.info("📊 Model loaded successfully")
            
            return model, tokenizer
            
        except ImportError:
            raise ModelError(
                "Unsloth not installed. Please install with: pip install unsloth"
            )
        except Exception as e:
            raise ModelError(f"Failed to create Unsloth model: {e}")
    
    @staticmethod
    def _create_huggingface_model(config: ConfigManager) -> Tuple[Any, Any]:
        """Create standard HuggingFace model with optional quantization."""
        try:
            logger.info(f"🤗 Creating HuggingFace model: {config.model.name}")
            
            # Setup quantization if requested
            quantization_config = None
            if config.model.load_in_4bit or config.model.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=config.model.load_in_4bit,
                    load_in_8bit=config.model.load_in_8bit,
                    bnb_4bit_quant_type="nf4" if config.model.load_in_4bit else None,
                    bnb_4bit_compute_dtype=torch.bfloat16 if config.model.load_in_4bit else None,
                    bnb_4bit_use_double_quant=True if config.model.load_in_4bit else None,
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.name,
                trust_remote_code=True,
                use_fast=True,
            )
            
            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if config.training.bf16 else torch.float16,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            )
            
            # Apply LoRA if not doing full fine-tuning
            if not config.model.full_finetuning:
                model = ModelFactory._apply_lora_to_hf_model(model, config)
            
            logger.info("📊 HuggingFace model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            raise ModelError(f"Failed to create HuggingFace model: {e}")
    
    @staticmethod
    def _apply_lora_to_hf_model(model: Any, config: ConfigManager) -> Any:
        """Apply LoRA to HuggingFace model using PEFT."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            logger.info("🔧 Applying LoRA to HuggingFace model...")
            
            peft_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.dropout,
                bias=config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                use_rslora=config.lora.use_rslora,
            )
            
            model = get_peft_model(model, peft_config)
            return model
            
        except ImportError:
            raise ModelError(
                "PEFT not installed. Please install with: pip install peft"
            )
        except Exception as e:
            raise ModelError(f"Failed to apply LoRA: {e}")
    
    @staticmethod
    def is_supported(model_name: str) -> bool:
        """Check if model is supported by factory."""
        return (ModelFactory._is_unsloth_model(model_name) or 
                ModelFactory._is_huggingface_model(model_name))
    
    @staticmethod
    def get_supported_models() -> dict:
        """Get list of supported model types."""
        return ModelFactory.SUPPORTED_MODEL_TYPES
    
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
