"""Inference engine for trained Gemma3N models."""

import torch
from typing import Dict, Any, List, Optional, Union
from training_pipeline.utils import ChatFormatter
from training_pipeline.config import GenerationConfigSection


class InferenceEngine:
    """Handles model inference with various generation options."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        generation_config: GenerationConfigSection,
        device: str = "cuda",):
        """
        Initialize InferenceEngine.
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer/processor
            device: Device for inference
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            do_sample: Whether to use sampling
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chat_formatter = ChatFormatter(tokenizer)
        self.generation_config = generation_config.copy()
        # Generation parameters
        self.generation_config["pad_token_id"] = tokenizer.eos_token_id,
        
        # Prepare model for inference
        self._setup_for_inference()
    
    def _setup_for_inference(self) -> None:
        """Setup model for inference."""
        try:
            # Use Unsloth optimization if available
            from unsloth import FastModel
            FastModel.for_inference(self.model)
            print("🚀 Model optimized for inference with Unsloth")
        except:
            # Fallback to standard setup
            self.model.eval()
            print("📝 Model set to evaluation mode")
        
        # Move to device if needed
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
    
    def generate(
        self,
        question: str,
        generation_config: Optional[Dict[str, Any]] = None,
        return_full_text: bool = False
    ) -> str:
        """
        Generate response for a question.
        
        Args:
            question: Input question
            generation_config: Optional generation config overrides
            return_full_text: Whether to return full text including prompt
            
        Returns:
            Generated response text
        """
        # Prepare inputs
        inputs = self.chat_formatter.prepare_inference_inputs(
            question=question,
            device=self.device
        )
        
        # Update generation config
        gen_config = self.generation_config.copy()
        if generation_config:
            gen_config.update(generation_config)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
        
        # Decode outputs
        if return_full_text:
            # Return full generated text
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            # Return only the new tokens (response)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def generate_batch(
        self,
        questions: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 4,
        return_full_text: bool = False
    ) -> List[str]:
        """
        Generate responses for multiple questions.
        
        Args:
            questions: List of input questions
            generation_config: Optional generation config overrides
            batch_size: Batch size for processing
            return_full_text: Whether to return full text including prompt
            
        Returns:
            List of generated responses
        """
        responses = []
        
        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            
            # Generate for each question in batch
            # Note: For simplicity, processing one by one
            # Can be optimized for true batch processing
            for question in batch_questions:
                response = self.generate(
                    question=question,
                    generation_config=generation_config,
                    return_full_text=return_full_text
                )
                responses.append(response)
        
        return responses
    
    def generate_with_streaming(
        self,
        question: str,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response with streaming output.
        
        Args:
            question: Input question
            generation_config: Optional generation config overrides
            
        Returns:
            Generated response text
        """
        try:
            from transformers import TextStreamer
            
            # Prepare inputs
            inputs = self.chat_formatter.prepare_inference_inputs(
                question=question,
                device=self.device
            )
            
            # Setup streamer
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # Update generation config
            gen_config = self.generation_config.copy()
            if generation_config:
                gen_config.update(generation_config)
            gen_config["streamer"] = streamer
            
            # Generate with streaming
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
            
            # Decode the full output for return
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except ImportError:
            print("Warning: TextStreamer not available, falling back to regular generation")
            return self.generate(question, generation_config)
    
    def update_generation_config(self, **kwargs) -> None:
        """
        Update generation configuration.
        
        Args:
            **kwargs: Generation config parameters to update
        """
        self.generation_config.update(kwargs)
        print(f"🔧 Generation config updated: {kwargs}")
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get current generation configuration.
        
        Returns:
            Current generation configuration
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