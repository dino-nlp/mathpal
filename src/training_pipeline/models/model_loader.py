"""Model loading utilities using Unsloth."""

from typing import Tuple, Optional, Dict, Any
import torch
from unsloth import FastModel, get_chat_template
from ..config.config_manager import ConfigManager


class ModelLoader:
    """Handles model and processor loading with Unsloth optimizations."""
    
    def __init__(self, config: ConfigManager):
        """Initialize ModelLoader with training configuration."""
        self.config = config
        
    def load_model_and_processor(self) -> Tuple[Any, Any]:
        """
        Load and setup Gemma3N model and processor using Unsloth.
        
        Returns:
            Tuple of (model, processor)
        """
        print("Loading Gemma3N model and processor...")
        
        try:
            # Load model and processor with Unsloth
            model, processor = FastModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                full_finetuning=self.config.full_finetuning,
            )
            
            # Setup chat template for Gemma3N
            processor = get_chat_template(processor, "gemma-3n")
            
            print("Model and processor setup complete!")
            return model, processor
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def apply_lora(
        self, 
        model: Any,
        lora_config_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Apply LoRA adapters to the model.
        
        Args:
            model: Base model to apply LoRA to
            lora_config_kwargs: Optional LoRA configuration overrides
            
        Returns:
            Model with LoRA adapters applied
        """
        print("Applying LoRA adapters...")
        
        # Get LoRA configuration
        lora_kwargs = self.config.to_lora_config_kwargs()
        if lora_config_kwargs:
            lora_kwargs.update(lora_config_kwargs)
        
        try:
            # Apply LoRA using Unsloth
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                **lora_kwargs
            )
            
            print("LoRA adapters applied successfully!")
            return model
            
        except Exception as e:
            print(f"Error applying LoRA: {e}")
            raise
    
    def setup_model_for_training(self, model: Any) -> Any:
        """
        Prepare model for training using Unsloth optimizations.
        
        Args:
            model: Model to prepare for training
            
        Returns:
            Model ready for training
        """
        try:
            # Enable training mode with Unsloth optimizations
            FastModel.for_training(model)
            print("Model prepared for training!")
            return model
            
        except Exception as e:
            print(f"Error preparing model for training: {e}")
            raise
    
    def setup_model_for_inference(self, model: Any) -> Any:
        """
        Prepare model for inference using Unsloth optimizations.
        
        Args:
            model: Model to prepare for inference
            
        Returns:
            Model ready for inference
        """
        try:
            # Enable inference mode with Unsloth optimizations
            FastModel.for_inference(model)
            print("Model prepared for inference!")
            return model
            
        except Exception as e:
            print(f"Error preparing model for inference: {e}")
            raise
    
    def load_complete_model(
        self, 
        apply_lora: bool = True,
        lora_config_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Any]:
        """
        Load complete model with all configurations applied.
        
        Args:
            apply_lora: Whether to apply LoRA adapters
            lora_config_kwargs: Optional LoRA configuration overrides
            
        Returns:
            Tuple of (configured_model, processor)
        """
        # Load base model and processor
        model, processor = self.load_model_and_processor()
        
        # Apply LoRA if requested
        if apply_lora:
            model = self.apply_lora(model, lora_config_kwargs)
        
        return model, processor
    
    def print_model_info(self, model: Any) -> None:
        """
        Print information about the loaded model.
        
        Args:
            model: Model to analyze
        """
        try:
            # Print trainable parameters if available
            if hasattr(model, 'print_trainable_parameters'):
                print("\n📊 Model Information:")
                model.print_trainable_parameters()
            else:
                # Count parameters manually
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"\n📊 Model Information:")
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Trainable%: {100 * trainable_params / total_params:.2f}%")
                
        except Exception as e:
            print(f"Error getting model info: {e}")
    
    @staticmethod
    def get_model_memory_usage() -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        return {"message": "CUDA not available"}