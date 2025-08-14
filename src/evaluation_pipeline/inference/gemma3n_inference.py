"""
Gemma 3N inference engine for evaluation pipeline.
Follows the pattern from training_pipeline/inference/inference_engine.py
"""

import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)

from ..utils import (
    InferenceError,
    ModelError,
    get_logger,
    get_device_info
)


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    
    name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    batch_size: int = 8
    torch_dtype: str = "float16"
    device_map: str = "auto"


class ChatFormatter:
    """Simple chat formatter for evaluation pipeline."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        """Initialize chat formatter."""
        self.tokenizer = tokenizer
    
    def prepare_inference_inputs(self, question: str, device: str = "cuda") -> Dict[str, Any]:
        """Prepare inputs using chat template when available."""
        # Prefer chat template if tokenizer supports it
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            )
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            return inputs
        
        # Fallback to plain encoding
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        return inputs


class Gemma3NInferenceEngine:
    """
    Gemma 3N inference engine for evaluation pipeline.
    
    Follows the pattern from training_pipeline/inference/inference_engine.py
    """
    
    def __init__(self, model_config: ModelConfig, hardware_config: Dict[str, Any]):
        """
        Initialize Gemma 3N inference engine.
        
        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
        """
        self.model_config = model_config
        self.hardware_config = hardware_config
        self.logger = get_logger("Gemma3NInferenceEngine")
        
        # Setup device
        self.device = self._setup_device()
        
        # Model and tokenizer will be loaded when needed
        self.model = None
        self.tokenizer = None
        self.chat_formatter = None
        
        # Default generation config
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None  # Will be set when tokenizer is loaded
        }
        
        self.logger.info("Gemma 3N inference engine initialized")
    
    def _setup_device(self) -> str:
        """Setup device for inference."""
        device_info = get_device_info()
        
        if device_info["cuda_available"]:
            device = "cuda"
            self.logger.info(f"Using CUDA device: {device_info['device_name']}")
        else:
            device = "cpu"
            self.logger.info("Using CPU device")
        
        return device
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load Gemma 3N model with optimizations.
        
        Args:
            model_path: Path to the model
            
        Raises:
            ModelError: If model cannot be loaded
        """
        try:
            self.logger.info(f"Loading Gemma 3N model from {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize chat formatter
            self.chat_formatter = ChatFormatter(self.tokenizer)
            
            # Prepare model kwargs with only valid parameters
            model_kwargs = self._prepare_model_kwargs()
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Setup model for inference
            self._setup_for_inference()
            
            # Update generation config with tokenizer info
            self.generation_config["pad_token_id"] = self.tokenizer.eos_token_id
            
            self.logger.info("Model loaded successfully")
            self._log_model_info()
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise ModelError(f"Error loading Gemma 3N model: {e}")
    
    def _prepare_model_kwargs(self) -> Dict[str, Any]:
        """
        Prepare keyword arguments for model loading.
        
        Returns:
            Model loading configuration with only valid parameters
        """
        model_kwargs = {
            "torch_dtype": getattr(torch, self.model_config.torch_dtype),
            "device_map": self.model_config.device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        # Add quantization if specified
        if self.model_config.load_in_4bit:
            self.logger.info("Using 4-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.model_config.load_in_8bit:
            self.logger.info("Using 8-bit quantization")
            model_kwargs["load_in_8bit"] = True
        
        return model_kwargs
    
    def _setup_for_inference(self) -> None:
        """Setup model for inference."""
        # Use standard setup without Unsloth to avoid compatibility issues
        self.model.eval()
        self.logger.info("Model set to evaluation mode")
        
        # Move to device if needed (skip for quantized models)
        if hasattr(self.model, 'to') and not self._is_quantized():
            self.model = self.model.to(self.device)
    
    def _is_quantized(self) -> bool:
        """Check if model is quantized."""
        return (
            hasattr(self.model, 'is_loaded_in_8bit') and self.model.is_loaded_in_8bit
        ) or (
            hasattr(self.model, 'is_loaded_in_4bit') and self.model.is_loaded_in_4bit
        ) or (
            self.model_config.load_in_4bit or self.model_config.load_in_8bit
        )
    
    def _log_model_info(self) -> None:
        """Log model information."""
        if self.model is None:
            return
        
        # Get model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get model size
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        self.logger.info(f"Model loaded successfully:")
        self.logger.info(f"  - Total parameters: {total_params:,}")
        self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  - Model size: {model_size_mb:.2f} MB")
        self.logger.info(f"  - Device: {self.device}")
        self.logger.info(f"  - Quantization: {'4-bit' if self.model_config.load_in_4bit else '8-bit' if self.model_config.load_in_8bit else 'None'}")
    
    def generate(
        self,
        question: str,
        generation_config: Optional[Dict[str, Any]] = None,
        return_full_text: bool = False
    ) -> str:
        """
        Generate response for a single question.
        
        Args:
            question: Input question
            generation_config: Optional generation config overrides
            return_full_text: Whether to return full text including prompt
            
        Returns:
            Generated response text
        """
        # Use batch processing with single item for consistency
        responses = self.generate_batch(
            questions=[question],
            generation_config=generation_config,
            batch_size=1,
            return_full_text=return_full_text
        )
        
        return responses[0] if responses else ""
    
    def generate_batch(
        self,
        questions: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
        batch_size: int = None,
        return_full_text: bool = False
    ) -> List[str]:
        """
        Generate responses for multiple questions using true batch processing.
        
        Args:
            questions: List of input questions
            generation_config: Optional generation config overrides
            batch_size: Batch size for processing (defaults to model_config.batch_size)
            return_full_text: Whether to return full text including prompt
            
        Returns:
            List of generated responses
        """
        if self.model is None or self.tokenizer is None:
            raise InferenceError("Model not loaded. Call load_model() first.")
        
        if not questions:
            return []
        
        batch_size = batch_size or self.model_config.batch_size
        responses = []
        
        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            
            # Prepare batch inputs
            batch_inputs = []
            for question in batch_questions:
                inputs = self.chat_formatter.prepare_inference_inputs(
                    question=question,
                    device=self.device
                )
                batch_inputs.append(inputs)
            
            # Stack inputs for batch processing
            if len(batch_inputs) == 1:
                # Single item - use as is
                batch_inputs = batch_inputs[0]
            else:
                # Multiple items - pad and stack
                batch_inputs = self._prepare_batch_inputs(batch_inputs)
            
            # Update generation config
            gen_config = self.generation_config.copy()
            if generation_config:
                gen_config.update(generation_config)
            
            # Generate for batch
            with torch.no_grad():
                batch_outputs = self.model.generate(**batch_inputs, **gen_config)
            
            # Decode batch outputs
            batch_responses = self._decode_batch_outputs(
                batch_outputs, 
                batch_inputs, 
                return_full_text
            )
            
            responses.extend(batch_responses)
        
        return responses
    
    def _prepare_batch_inputs(self, batch_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare batched inputs for model generation.
        
        Args:
            batch_inputs: List of input dictionaries
            
        Returns:
            Batched input dictionary
        """
        if len(batch_inputs) == 1:
            return batch_inputs[0]
        
        # Stack input_ids and attention_mask
        input_ids = torch.stack([inputs['input_ids'] for inputs in batch_inputs])
        attention_mask = torch.stack([inputs['attention_mask'] for inputs in batch_inputs])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def _decode_batch_outputs(
        self, 
        batch_outputs: torch.Tensor, 
        batch_inputs: Dict[str, Any], 
        return_full_text: bool
    ) -> List[str]:
        """
        Decode batch outputs to text responses.
        
        Args:
            batch_outputs: Model outputs tensor
            batch_inputs: Input dictionary used for generation
            return_full_text: Whether to return full text
            
        Returns:
            List of decoded text responses
        """
        responses = []
        
        if return_full_text:
            # Return full generated text for each item
            for i in range(batch_outputs.shape[0]):
                generated_text = self.tokenizer.decode(
                    batch_outputs[i], 
                    skip_special_tokens=True
                )
                responses.append(generated_text.strip())
        else:
            # Return only new tokens for each item
            input_length = batch_inputs['input_ids'].shape[1]
            for i in range(batch_outputs.shape[0]):
                new_tokens = batch_outputs[i][input_length:]
                generated_text = self.tokenizer.decode(
                    new_tokens, 
                    skip_special_tokens=True
                )
                responses.append(generated_text.strip())
        
        return responses
    
    def update_generation_config(self, **kwargs) -> None:
        """
        Update generation configuration.
        
        Args:
            **kwargs: Generation config parameters to update
        """
        self.generation_config.update(kwargs)
        self.logger.info(f"Generation config updated: {kwargs}")
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get current generation configuration.
        
        Returns:
            Current generation configuration (copy of the dict)
        """
        return self.generation_config.copy()
    
    @staticmethod
    def get_recommended_configs() -> Dict[str, Dict[str, Any]]:
        """
        Get recommended generation configurations.
        
        Returns:
            Dictionary of configuration presets
        """
        return {
            "creative": {
                "temperature": 1.2,
                "top_p": 0.9,
                "top_k": 50,
                "max_new_tokens": 128
            },
            "balanced": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "max_new_tokens": 64
            },
            "focused": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_new_tokens": 64
            },
            "deterministic": {
                "temperature": 0.1,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": 64,
                "do_sample": False
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up inference engine resources...")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared")
            
            # Unload model
            if self.model is not None:
                del self.model
                self.model = None
                self.logger.info("Model unloaded")
            
            # Clear tokenizer
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear chat formatter
            if self.chat_formatter is not None:
                del self.chat_formatter
                self.chat_formatter = None
            
            self.logger.info("Inference engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors in destructor
            pass
