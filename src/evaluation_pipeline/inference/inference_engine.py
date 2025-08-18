"""Inference engine for trained Gemma3N models."""

import torch
from typing import Dict, Any, List, Optional, Union
from ..config import ConfigManager
from ..utils import get_logger

class InferenceEngine:
    """Handles model inference with various generation options."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config_manager: ConfigManager,
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
        self.generation_config = config_manager.get_generation_config()
        # Generation parameters
        self.generation_config["pad_token_id"] = tokenizer.eos_token_id
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    
    def _format_inference_input(self, question: str) -> List[Dict[str, Any]]:
        """
        Format a single question for inference.
        
        Args:
            question: Input question text
            
        Returns:
            Formatted messages for chat template
        """
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": question,
            }]
        }]
    
    def _prepare_inference_input(self, question: str, device: str = "cuda") -> Dict[str, Any]:
        """
        Prepare inputs for model inference.
        
        Args:
            question: Input question
            device: Target device
            
        Returns:
            Tokenized inputs ready for model
        """
        messages = self._format_inference_input(question)
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to(device)
        
        return inputs
    
    def generate(
        self,
        question: str,
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
            batch_size=1,
            return_full_text=return_full_text
        )
        
        return responses[0] if responses else ""
    
    def generate_batch(
        self,
        questions: List[str],
        batch_size: int = 4,
        return_full_text: bool = False
    ) -> List[str]:
        """
        Generate responses for multiple questions using true batch processing.
        
        Args:
            questions: List of input questions
            generation_config: Optional generation config overrides
            batch_size: Batch size for processing
            return_full_text: Whether to return full text including prompt
            
        Returns:
            List of generated responses
        """
        if not questions:
            return []
        
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
            Current generation configuration (copy of the dict)
        """
        return self.generation_config.copy()
    

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