"""Inference engine for trained Gemma3N models."""

import torch
from typing import Dict, Any, List, Optional, Union
from training_pipeline.data.chat_formatter import ChatFormatter


class InferenceEngine:
    """Handles model inference with various generation options."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        do_sample: bool = True
    ):
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
        
        # Generation parameters
        self.generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
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
    
    def benchmark_inference(
        self,
        questions: List[str],
        num_runs: int = 3,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            questions: List of test questions
            num_runs: Number of runs for averaging
            generation_config: Optional generation config
            
        Returns:
            Benchmark results
        """
        import time
        
        results = {
            "num_questions": len(questions),
            "num_runs": num_runs,
            "times": [],
            "tokens_generated": [],
            "tokens_per_second": []
        }
        
        print(f"🏃‍♂️ Benchmarking inference on {len(questions)} questions, {num_runs} runs...")
        
        for run in range(num_runs):
            start_time = time.time()
            total_tokens = 0
            
            for question in questions:
                response = self.generate(
                    question=question,
                    generation_config=generation_config
                )
                # Handle different tokenizer types
                if response and isinstance(response, str):
                    try:
                        # Try standard tokenizer encode method
                        total_tokens += len(self.tokenizer.encode(response))
                    except AttributeError:
                        # For processors like Gemma3nProcessor, use __call__ method
                        try:
                            tokenized = self.tokenizer(text=response, return_tensors="pt", add_special_tokens=False)
                            total_tokens += tokenized['input_ids'].shape[1]
                        except Exception as e:
                            # Fallback: estimate tokens by word count
                            print(f"Warning: Could not tokenize response, using word count estimate: {e}")
                            total_tokens += len(response.split())
                else:
                    print(f"Warning: Invalid response received: {response}")
                    total_tokens += 0
            
            end_time = time.time()
            run_time = end_time - start_time
            
            results["times"].append(run_time)
            results["tokens_generated"].append(total_tokens)
            results["tokens_per_second"].append(total_tokens / run_time)
            
            print(f"   Run {run + 1}: {run_time:.2f}s, {total_tokens} tokens, {total_tokens/run_time:.1f} tok/s")
        
        # Calculate averages
        results["avg_time"] = sum(results["times"]) / num_runs
        results["avg_tokens"] = sum(results["tokens_generated"]) / num_runs
        results["avg_tokens_per_second"] = sum(results["tokens_per_second"]) / num_runs
        
        print(f"📊 Average: {results['avg_time']:.2f}s, {results['avg_tokens_per_second']:.1f} tok/s")
        
        return results
    
    def test_model(
        self,
        test_questions: Optional[List[str]] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Test model with sample questions.
        
        Args:
            test_questions: Optional list of test questions
            generation_config: Optional generation config
            
        Returns:
            List of question-answer pairs
        """
        if test_questions is None:
            test_questions = [
                "Tính tổng của 15 + 27 = ?",
                "Một hình chữ nhật có chiều dài 8m và chiều rộng 5m. Tính diện tích?",
                "Trong một lớp có 24 học sinh, trong đó có 13 học sinh nam. Có bao nhiêu học sinh nữ?",
                "Giải phương trình: 2x + 5 = 13"
            ]
        
        results = []
        print(f"🧪 Testing model with {len(test_questions)} questions...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {question}")
            
            response = self.generate(
                question=question,
                generation_config=generation_config
            )
            
            print(f"A: {response}")
            
            results.append({
                "question": question,
                "answer": response
            })
        
        return results
    
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