"""
Gemma 3N inference engine with MatFormer optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import time
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation import GenerationConfig

# Prefer Unsloth for Gemma-3N loading (supports gemma3n arch)
try:
    from unsloth import FastModel, get_chat_template  # type: ignore
    _HAS_UNSLOTH = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_UNSLOTH = False

from ..config import ConfigManager
from ..utils import (
    ModelError,
    InferenceError,
    get_logger,
    get_device_info,
    format_memory_size
)
from .matformer_utils import MatFormerOptimizer, apply_matformer_optimizations


class Gemma3NInferenceEngine:
    """
    Gemma 3N inference engine with MatFormer optimization.
    
    Provides optimized inference for Gemma 3N models with:
    - MatFormer attention optimization
    - Memory-efficient loading
    - Batch processing
    - Streaming support
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Gemma 3N inference engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger("Gemma3NInferenceEngine")
        
        # Model configuration
        self.model_config = config.get_model_config()
        self.hardware_config = config.get_hardware_config()
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Device setup
        self.device = self._setup_device()
        
        # MatFormer optimizer
        self.matformer_optimizer = MatFormerOptimizer(self.model_config.matformer_config)
        
        # Performance tracking
        self.inference_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "avg_tokens_per_second": 0.0
        }
        
        self.logger.info("Gemma 3N inference engine initialized")
    
    def _setup_device(self) -> torch.device:
        """
        Setup device for inference.
        
        Returns:
            Device to use for inference
        """
        device_info = get_device_info()
        
        if device_info["cuda_available"]:
            device = torch.device("cuda")
            self.logger.info(f"Using CUDA device: {device_info['device_name']}")
            
            # Set memory fraction if specified
            memory_fraction = self.hardware_config.get("memory_fraction", 0.9)
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Enable gradient checkpointing for memory efficiency
            if self.hardware_config.get("gradient_checkpointing", True):
                torch.backends.cudnn.benchmark = True
            
        else:
            device = torch.device("cpu")
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
            # Prefer Unsloth loader for Gemma-3N checkpoints (avoids unknown arch errors)
            if _HAS_UNSLOTH:
                self.logger.info("Loading Gemma 3N with Unsloth FastModel...")
                # FastModel.from_pretrained returns (model, tokenizer)
                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name=str(model_path),
                    dtype=None,
                    max_seq_length=self.model_config.max_seq_length,
                    load_in_4bit=self.model_config.load_in_4bit,
                    load_in_8bit=self.model_config.load_in_8bit,
                )
                # Ensure chat template is set for Gemma-3N
                try:
                    self.tokenizer = get_chat_template(self.tokenizer, "gemma-3n")
                except Exception as e:
                    self.logger.warning(f"Failed to set chat template via Unsloth: {e}")

                if getattr(self.tokenizer, "pad_token", None) is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Fallback to Transformers loader
                self.logger.info("Unsloth not available, falling back to Transformers loader...")
                model_path = Path(model_path)
                self.logger.info(f"Loading Gemma 3N model from {model_path}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                model_kwargs = self._prepare_model_kwargs()
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )

                if self.model_config.use_matformer:
                    self.logger.info("Applying MatFormer optimizations...")
                    self.model = apply_matformer_optimizations(self.model, self.model_config.matformer_config)

            # Move model to device and eval
            if self.device.type == "cuda" and hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

            # Optional compile for extra speed
            if hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    self.logger.info("Compiling model with torch.compile...")
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")

            self.logger.info("Gemma 3N model loaded successfully")
            self._log_model_info()

        except Exception as e:
            raise ModelError(f"Error loading Gemma 3N model: {e}")

    def _prepare_inference_inputs(self, question: str) -> Dict[str, Any]:
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
            ).to(self.device)
            return inputs
        # Fallback to plain encoding
        return self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_seq_length,
        ).to(self.device)
    
    def _prepare_model_kwargs(self) -> Dict[str, Any]:
        """
        Prepare keyword arguments for model loading.
        
        Returns:
            Model loading configuration
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
        
        # Add MatFormer configuration
        if self.model_config.use_matformer:
            matformer_config = self.matformer_optimizer.create_matformer_config()
            model_kwargs.update(matformer_config)
        
        # Add memory optimization configuration
        memory_config = self.matformer_optimizer.get_memory_optimization_config(
            "cuda" if self.device.type == "cuda" else "cpu"
        )
        model_kwargs.update(memory_config)
        
        return model_kwargs
    
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
        self.logger.info(f"  - MatFormer enabled: {self.model_config.use_matformer}")
        self.logger.info(f"  - Quantization: {'4-bit' if self.model_config.load_in_4bit else '8-bit' if self.model_config.load_in_8bit else 'None'}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate response for a single prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Raises:
            InferenceError: If generation fails
        """
        if self.model is None or self.tokenizer is None:
            raise InferenceError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Prepare inputs (chat template aware)
            inputs = self._prepare_inference_inputs(prompt)
            
            # Set generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Update statistics
            self._update_stats(start_time, len(outputs[0]) - inputs['input_ids'].shape[1])
            
            return response.strip()
            
        except Exception as e:
            raise InferenceError(f"Error generating response: {e}")
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
            
        Raises:
            InferenceError: If generation fails
        """
        if self.model is None or self.tokenizer is None:
            raise InferenceError("Model not loaded. Call load_model() first.")
        
        responses = []
        total_start_time = time.time()
        
        try:
            # Process in batches
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_start_time = time.time()
                
            # Prepare inputs for batch: build per-item then stack if needed
            batch_inputs_list: List[Dict[str, Any]] = [
                self._prepare_inference_inputs(q) for q in batch_prompts
            ]
            if len(batch_inputs_list) == 1:
                inputs = batch_inputs_list[0]
            else:
                # Stack input_ids and attention_mask
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [bi["input_ids"].squeeze(0) for bi in batch_inputs_list],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    [bi["attention_mask"].squeeze(0) for bi in batch_inputs_list],
                    batch_first=True,
                    padding_value=0,
                )
                inputs = {"input_ids": input_ids.to(self.device), "attention_mask": attention_mask.to(self.device)}
                
                # Set generation parameters
                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Decode responses
                for j, output in enumerate(outputs):
                    response = self.tokenizer.decode(
                        output[inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    responses.append(response.strip())
                
                # Update statistics
                total_new_tokens = sum(len(output) - inputs['input_ids'].shape[1] for output in outputs)
                self._update_stats(batch_start_time, total_new_tokens)
                
                # Clear cache if using CUDA
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            total_time = time.time() - total_start_time
            self.logger.info(f"Batch generation completed: {len(prompts)} prompts in {total_time:.2f}s")
            
            return responses
            
        except Exception as e:
            raise InferenceError(f"Error in batch generation: {e}")
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ):
        """
        Generate response with streaming output.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Yields:
            Generated tokens as they are produced
        """
        if self.model is None or self.tokenizer is None:
            raise InferenceError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config.max_seq_length
            ).to(self.device)
            
            # Set generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Generate with streaming
            with torch.no_grad():
                for outputs in self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    streamer=None,  # We'll handle streaming manually
                    return_dict_in_generate=True,
                    output_scores=False,
                    **kwargs
                ):
                    # Decode new tokens
                    new_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
                    if len(new_tokens) > 0:
                        new_text = self.tokenizer.decode(new_tokens[-1:], skip_special_tokens=True)
                        yield new_text
            
            # Update statistics
            total_tokens = len(outputs.sequences[0]) - inputs['input_ids'].shape[1]
            self._update_stats(start_time, total_tokens)
            
        except Exception as e:
            raise InferenceError(f"Error in streaming generation: {e}")
    
    def _update_stats(self, start_time: float, new_tokens: int) -> None:
        """
        Update inference statistics.
        
        Args:
            start_time: Start time of inference
            new_tokens: Number of new tokens generated
        """
        inference_time = time.time() - start_time
        
        self.inference_stats["total_requests"] += 1
        self.inference_stats["total_tokens"] += new_tokens
        self.inference_stats["total_time"] += inference_time
        
        # Update average tokens per second
        if self.inference_stats["total_time"] > 0:
            self.inference_stats["avg_tokens_per_second"] = (
                self.inference_stats["total_tokens"] / self.inference_stats["total_time"]
            )
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """
        Get inference statistics.
        
        Returns:
            Dictionary with inference statistics
        """
        stats = self.inference_stats.copy()
        
        # Add memory information
        if self.device.type == "cuda":
            stats["gpu_memory_allocated"] = format_memory_size(torch.cuda.memory_allocated())
            stats["gpu_memory_reserved"] = format_memory_size(torch.cuda.memory_reserved())
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("GPU cache cleared")
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.clear_cache()
        self.logger.info("Model unloaded")
