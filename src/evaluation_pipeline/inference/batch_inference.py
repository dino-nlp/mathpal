"""
Batch inference engine for processing large batches efficiently.
"""

import torch
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..utils import (
    BatchProcessingError,
    get_logger,
    format_memory_size
)
from .gemma3n_inference import Gemma3NInferenceEngine


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    
    batch_id: int
    prompts: List[str]
    responses: List[str]
    processing_time: float
    tokens_generated: int
    memory_used: str
    success: bool
    error_message: Optional[str] = None


class BatchInferenceEngine:
    """
    Batch inference engine for processing large batches efficiently.
    
    Features:
    - Dynamic batch sizing based on memory
    - Parallel processing with thread pools
    - Memory monitoring and optimization
    - Progress tracking and statistics
    """
    
    def __init__(self, inference_engine: Gemma3NInferenceEngine):
        """
        Initialize batch inference engine.
        
        Args:
            inference_engine: Gemma 3N inference engine
        """
        self.inference_engine = inference_engine
        self.logger = get_logger("BatchInferenceEngine")
        
        # Batch processing configuration
        self.max_batch_size = 8
        self.max_workers = 4
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
        # Statistics
        self.batch_stats = {
            "total_batches": 0,
            "total_prompts": 0,
            "total_responses": 0,
            "total_processing_time": 0.0,
            "total_tokens": 0,
            "failed_batches": 0
        }
        
        self.logger.info("Batch inference engine initialized")
    
    def process_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Process a batch of prompts efficiently.
        
        Args:
            prompts: List of prompts to process
            max_new_tokens: Maximum new tokens per response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
            
        Raises:
            BatchProcessingError: If batch processing fails
        """
        if not prompts:
            return []
        
        self.logger.info(f"Processing batch of {len(prompts)} prompts")
        
        # Determine optimal batch size
        optimal_batch_size = self._calculate_optimal_batch_size(len(prompts))
        self.logger.info(f"Using batch size: {optimal_batch_size}")
        
        # Split prompts into batches
        batches = self._split_into_batches(prompts, optimal_batch_size)
        
        # Process batches
        all_responses = []
        batch_results = []
        
        for i, batch_prompts in enumerate(batches):
            self.logger.debug(f"Processing batch {i+1}/{len(batches)}")
            
            try:
                # Process single batch
                batch_result = self._process_single_batch(
                    batch_id=i,
                    prompts=batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    **kwargs
                )
                
                batch_results.append(batch_result)
                all_responses.extend(batch_result.responses)
                
                # Update statistics
                self._update_batch_stats(batch_result)
                
                # Memory management
                self._manage_memory()
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i}: {e}")
                self.batch_stats["failed_batches"] += 1
                
                # Add placeholder responses for failed batch
                placeholder_responses = [f"Error: {str(e)}" for _ in batch_prompts]
                all_responses.extend(placeholder_responses)
        
        # Log final statistics
        self._log_batch_statistics()
        
        return all_responses
    
    def process_batch_parallel(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Process batches in parallel using thread pool.
        
        Args:
            prompts: List of prompts to process
            max_new_tokens: Maximum new tokens per response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if not prompts:
            return []
        
        self.logger.info(f"Processing {len(prompts)} prompts in parallel")
        
        # Determine optimal batch size
        optimal_batch_size = self._calculate_optimal_batch_size(len(prompts))
        
        # Split prompts into batches
        batches = self._split_into_batches(prompts, optimal_batch_size)
        
        # Process batches in parallel
        all_responses = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {}
            
            for i, batch_prompts in enumerate(batches):
                future = executor.submit(
                    self._process_single_batch,
                    batch_id=i,
                    prompts=batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    **kwargs
                )
                future_to_batch[future] = i
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_result = future.result()
                    
                    # Update responses in correct order
                    start_idx = batch_id * optimal_batch_size
                    for j, response in enumerate(batch_result.responses):
                        if start_idx + j < len(all_responses):
                            all_responses[start_idx + j] = response
                    
                    # Update statistics
                    self._update_batch_stats(batch_result)
                    
                except Exception as e:
                    self.logger.error(f"Error in parallel batch {batch_id}: {e}")
                    self.batch_stats["failed_batches"] += 1
                    
                    # Fill with error responses
                    start_idx = batch_id * optimal_batch_size
                    for j in range(len(batches[batch_id])):
                        if start_idx + j < len(all_responses):
                            all_responses[start_idx + j] = f"Error: {str(e)}"
        
        # Filter out None values
        all_responses = [r for r in all_responses if r is not None]
        
        self._log_batch_statistics()
        
        return all_responses
    
    def _process_single_batch(
        self,
        batch_id: int,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> BatchResult:
        """
        Process a single batch of prompts.
        
        Args:
            batch_id: Batch identifier
            prompts: List of prompts in the batch
            max_new_tokens: Maximum new tokens per response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Batch result
        """
        start_time = time.time()
        
        try:
            # Get memory usage before processing
            memory_before = self._get_memory_usage()
            
            # Process batch
            responses = self.inference_engine.batch_generate(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                batch_size=len(prompts),  # Process entire batch at once
                **kwargs
            )
            
            # Calculate statistics
            processing_time = time.time() - start_time
            tokens_generated = sum(len(response.split()) for response in responses)
            memory_after = self._get_memory_usage()
            
            self.logger.debug(f"Batch {batch_id} completed: {len(prompts)} prompts in {processing_time:.2f}s")
            
            return BatchResult(
                batch_id=batch_id,
                prompts=prompts,
                responses=responses,
                processing_time=processing_time,
                tokens_generated=tokens_generated,
                memory_used=memory_after,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Batch {batch_id} failed: {e}")
            
            return BatchResult(
                batch_id=batch_id,
                prompts=prompts,
                responses=[f"Error: {str(e)}" for _ in prompts],
                processing_time=processing_time,
                tokens_generated=0,
                memory_used=self._get_memory_usage(),
                success=False,
                error_message=str(e)
            )
    
    def _calculate_optimal_batch_size(self, total_prompts: int) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            total_prompts: Total number of prompts to process
            
        Returns:
            Optimal batch size
        """
        # Get current memory usage
        memory_usage = self._get_memory_usage_percentage()
        
        # Adjust batch size based on memory usage
        if memory_usage > self.memory_threshold:
            # Reduce batch size if memory usage is high
            optimal_size = max(1, self.max_batch_size // 2)
        else:
            # Use maximum batch size if memory is available
            optimal_size = self.max_batch_size
        
        # Ensure batch size doesn't exceed total prompts
        optimal_size = min(optimal_size, total_prompts)
        
        return optimal_size
    
    def _split_into_batches(self, prompts: List[str], batch_size: int) -> List[List[str]]:
        """
        Split prompts into batches.
        
        Args:
            prompts: List of prompts
            batch_size: Size of each batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _manage_memory(self) -> None:
        """Manage memory usage during batch processing."""
        memory_usage = self._get_memory_usage_percentage()
        
        if memory_usage > self.memory_threshold:
            self.logger.warning(f"High memory usage detected: {memory_usage:.1%}")
            
            # Clear cache
            self.inference_engine.clear_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Reduce batch size for next batch
            self.max_batch_size = max(1, self.max_batch_size // 2)
            self.logger.info(f"Reduced batch size to {self.max_batch_size}")
    
    def _get_memory_usage(self) -> str:
        """
        Get current memory usage.
        
        Returns:
            Memory usage as formatted string
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            return format_memory_size(allocated)
        else:
            return "N/A (CPU)"
    
    def _get_memory_usage_percentage(self) -> float:
        """
        Get memory usage as percentage.
        
        Returns:
            Memory usage percentage (0.0 to 1.0)
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        else:
            return 0.0
    
    def _update_batch_stats(self, batch_result: BatchResult) -> None:
        """
        Update batch processing statistics.
        
        Args:
            batch_result: Result of batch processing
        """
        self.batch_stats["total_batches"] += 1
        self.batch_stats["total_prompts"] += len(batch_result.prompts)
        self.batch_stats["total_responses"] += len(batch_result.responses)
        self.batch_stats["total_processing_time"] += batch_result.processing_time
        self.batch_stats["total_tokens"] += batch_result.tokens_generated
        
        if not batch_result.success:
            self.batch_stats["failed_batches"] += 1
    
    def _log_batch_statistics(self) -> None:
        """Log batch processing statistics."""
        stats = self.batch_stats
        
        if stats["total_batches"] > 0:
            avg_time_per_batch = stats["total_processing_time"] / stats["total_batches"]
            avg_time_per_prompt = stats["total_processing_time"] / stats["total_prompts"]
            success_rate = (stats["total_batches"] - stats["failed_batches"]) / stats["total_batches"]
            
            self.logger.info("Batch processing statistics:")
            self.logger.info(f"  - Total batches: {stats['total_batches']}")
            self.logger.info(f"  - Total prompts: {stats['total_prompts']}")
            self.logger.info(f"  - Total responses: {stats['total_responses']}")
            self.logger.info(f"  - Total processing time: {stats['total_processing_time']:.2f}s")
            self.logger.info(f"  - Total tokens generated: {stats['total_tokens']}")
            self.logger.info(f"  - Average time per batch: {avg_time_per_batch:.2f}s")
            self.logger.info(f"  - Average time per prompt: {avg_time_per_prompt:.2f}s")
            self.logger.info(f"  - Success rate: {success_rate:.1%}")
            self.logger.info(f"  - Failed batches: {stats['failed_batches']}")
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """
        Get batch processing statistics.
        
        Returns:
            Dictionary with batch statistics
        """
        return self.batch_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset batch processing statistics."""
        self.batch_stats = {
            "total_batches": 0,
            "total_prompts": 0,
            "total_responses": 0,
            "total_processing_time": 0.0,
            "total_tokens": 0,
            "failed_batches": 0
        }
        self.logger.info("Batch statistics reset")
    
    def set_batch_size(self, batch_size: int) -> None:
        """
        Set maximum batch size.
        
        Args:
            batch_size: New maximum batch size
        """
        self.max_batch_size = max(1, batch_size)
        self.logger.info(f"Batch size set to {self.max_batch_size}")
    
    def set_max_workers(self, max_workers: int) -> None:
        """
        Set maximum number of workers for parallel processing.
        
        Args:
            max_workers: New maximum number of workers
        """
        self.max_workers = max(1, max_workers)
        self.logger.info(f"Max workers set to {self.max_workers}")
    
    def set_memory_threshold(self, threshold: float) -> None:
        """
        Set memory usage threshold.
        
        Args:
            threshold: Memory threshold (0.0 to 1.0)
        """
        self.memory_threshold = max(0.1, min(0.95, threshold))
        self.logger.info(f"Memory threshold set to {self.memory_threshold:.1%}")
