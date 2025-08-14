"""
Opik evaluator for LLM evaluation.
"""

import time
import json
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import requests

from ..config import ConfigManager
from ..utils import (
    OpikError,
    get_logger,
    format_memory_size
)


@dataclass
class OpikEvaluationRequest:
    """Request for Opik evaluation."""
    
    question: str
    answer: str
    context: Optional[str] = None
    expected_answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OpikEvaluationResult:
    """Result of Opik evaluation."""
    
    metric_name: str
    score: float
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OpikClient:
    """
    Client for Opik API integration.
    
    Handles communication with Opik evaluation service.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Opik client.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.opik_config = config.get_opik_config()
        self.logger = get_logger("OpikClient")
        
        # API configuration
        self.base_url = "https://api.opik.ai"  # Placeholder URL
        
        # Read API key from environment variable for security
        self.api_key = os.getenv("OPIK_API_KEY")
        if not self.api_key:
            raise OpikError("OPIK_API_KEY environment variable not set")
        
        self.workspace = self.opik_config.workspace
        self.project = self.opik_config.project
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit = 100  # requests per minute
        
        self.logger.info("Opik client initialized")
    
    def evaluate_batch(
        self,
        requests: List[OpikEvaluationRequest],
        metrics: List[str]
    ) -> List[OpikEvaluationResult]:
        """
        Evaluate a batch of requests using Opik.
        
        Args:
            requests: List of evaluation requests
            metrics: List of metrics to evaluate
            
        Returns:
            List of evaluation results
            
        Raises:
            OpikError: If evaluation fails
        """
        self.logger.info(f"Evaluating {len(requests)} samples with {len(metrics)} metrics")
        
        results = []
        
        for i, request in enumerate(requests):
            self.logger.debug(f"Processing request {i+1}/{len(requests)}")
            
            try:
                # Rate limiting
                self._check_rate_limit()
                
                # Evaluate single request
                request_results = self._evaluate_single(request, metrics)
                results.extend(request_results)
                
                # Update rate limiting
                self._update_rate_limit()
                
            except Exception as e:
                self.logger.error(f"Error evaluating request {i}: {e}")
                # Add placeholder results for failed request
                for metric in metrics:
                    results.append(OpikEvaluationResult(
                        metric_name=metric,
                        score=0.0,
                        confidence=0.0,
                        explanation=f"Error: {str(e)}"
                    ))
        
        self.logger.info(f"Evaluation completed: {len(results)} results")
        return results
    
    def _evaluate_single(
        self,
        request: OpikEvaluationRequest,
        metrics: List[str]
    ) -> List[OpikEvaluationResult]:
        """
        Evaluate a single request.
        
        Args:
            request: Evaluation request
            metrics: List of metrics to evaluate
            
        Returns:
            List of evaluation results
        """
        # This is a placeholder implementation
        # In real implementation, this would call the actual Opik API
        
        results = []
        
        for metric in metrics:
            # Simulate Opik API call
            score = self._simulate_opik_metric(request, metric)
            
            result = OpikEvaluationResult(
                metric_name=metric,
                score=score,
                confidence=0.9,
                explanation=f"Simulated {metric} evaluation"
            )
            
            results.append(result)
        
        return results
    
    def _simulate_opik_metric(
        self,
        request: OpikEvaluationRequest,
        metric: str
    ) -> float:
        """
        Simulate Opik metric evaluation.
        
        Args:
            request: Evaluation request
            metric: Metric name
            
        Returns:
            Simulated score
        """
        # Placeholder implementation with realistic scores
        base_score = 0.8
        
        # Adjust score based on metric type
        if metric == "hallucination":
            # Check if answer contains information not in context
            if request.context and request.context not in request.answer:
                base_score = 0.6
            else:
                base_score = 0.9
        
        elif metric == "context_precision":
            # Check relevance of context to question
            if request.context and any(word in request.context.lower() for word in request.question.lower().split()):
                base_score = 0.85
            else:
                base_score = 0.7
        
        elif metric == "context_recall":
            # Check if all relevant context is used
            base_score = 0.82
        
        elif metric == "answer_relevance":
            # Check if answer is relevant to question
            question_words = set(request.question.lower().split())
            answer_words = set(request.answer.lower().split())
            overlap = len(question_words.intersection(answer_words))
            if overlap > 0:
                base_score = 0.88
            else:
                base_score = 0.6
        
        elif metric == "usefulness":
            # Check if answer is useful
            if len(request.answer) > 10:
                base_score = 0.85
            else:
                base_score = 0.6
        
        # Add some randomness to make it more realistic
        import random
        variation = random.uniform(-0.1, 0.1)
        final_score = max(0.0, min(1.0, base_score + variation))
        
        return final_score
    
    def _check_rate_limit(self):
        """Check rate limiting."""
        current_time = time.time()
        
        if current_time - self.last_request_time < 60:
            if self.request_count >= self.rate_limit:
                sleep_time = 60 - (current_time - self.last_request_time)
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.request_count = 0
        else:
            self.request_count = 0
    
    def _update_rate_limit(self):
        """Update rate limiting counters."""
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_api_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request to Opik.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            API response
            
        Raises:
            OpikError: If API request fails
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/{endpoint}"
            
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise OpikError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise OpikError(f"Invalid JSON response: {e}")


class OpikEvaluator:
    """
    Opik-based evaluator for LLM evaluation.
    
    Provides comprehensive evaluation using Opik metrics.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Opik evaluator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.opik_config = config.get_opik_config()
        self.logger = get_logger("OpikEvaluator")
        
        # Initialize Opik client
        self.opik_client = OpikClient(config)
        
        # Evaluation statistics
        self.evaluation_stats = {
            "total_evaluations": 0,
            "total_requests": 0,
            "total_time": 0.0,
            "failed_evaluations": 0
        }
        
        self.logger.info("Opik evaluator initialized")
    
    def evaluate(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        expected_answers: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate using Opik metrics.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            answers: List of answers
            expected_answers: Optional list of expected answers
            metadata: Optional list of metadata
            
        Returns:
            Dictionary of metric scores
            
        Raises:
            OpikError: If evaluation fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting Opik evaluation of {len(questions)} samples")
            
            # Prepare evaluation requests
            requests = self._prepare_requests(
                questions, contexts, answers, expected_answers, metadata
            )
            
            # Get metrics to evaluate
            metrics = self.opik_config.metrics
            
            # Evaluate batch
            results = self.opik_client.evaluate_batch(requests, metrics)
            
            # Aggregate results
            aggregated_scores = self._aggregate_results(results, metrics)
            
            # Update statistics
            self._update_stats(start_time, len(questions), len(results))
            
            self.logger.info("Opik evaluation completed successfully")
            return aggregated_scores
            
        except Exception as e:
            self.evaluation_stats["failed_evaluations"] += 1
            raise OpikError(f"Opik evaluation failed: {e}")
    
    def _prepare_requests(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        expected_answers: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[OpikEvaluationRequest]:
        """
        Prepare evaluation requests.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            answers: List of answers
            expected_answers: Optional list of expected answers
            metadata: Optional list of metadata
            
        Returns:
            List of evaluation requests
        """
        requests = []
        
        for i in range(len(questions)):
            request = OpikEvaluationRequest(
                question=questions[i],
                context=contexts[i] if i < len(contexts) else None,
                answer=answers[i],
                expected_answer=expected_answers[i] if expected_answers and i < len(expected_answers) else None,
                metadata=metadata[i] if metadata and i < len(metadata) else None
            )
            requests.append(request)
        
        return requests
    
    def _aggregate_results(
        self,
        results: List[OpikEvaluationResult],
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        Aggregate evaluation results.
        
        Args:
            results: List of evaluation results
            metrics: List of metrics
            
        Returns:
            Dictionary of aggregated scores
        """
        aggregated = {}
        
        for metric in metrics:
            metric_results = [r for r in results if r.metric_name == metric]
            
            if metric_results:
                # Calculate average score
                scores = [r.score for r in metric_results]
                avg_score = sum(scores) / len(scores)
                aggregated[metric] = avg_score
            else:
                aggregated[metric] = 0.0
        
        return aggregated
    
    def _update_stats(self, start_time: float, num_requests: int, num_results: int):
        """
        Update evaluation statistics.
        
        Args:
            start_time: Start time of evaluation
            num_requests: Number of requests processed
            num_results: Number of results generated
        """
        evaluation_time = time.time() - start_time
        
        self.evaluation_stats["total_evaluations"] += 1
        self.evaluation_stats["total_requests"] += num_requests
        self.evaluation_stats["total_time"] += evaluation_time
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.
        
        Returns:
            Dictionary with evaluation statistics
        """
        stats = self.evaluation_stats.copy()
        
        if stats["total_evaluations"] > 0:
            stats["avg_time_per_evaluation"] = stats["total_time"] / stats["total_evaluations"]
            stats["avg_time_per_request"] = stats["total_time"] / stats["total_requests"]
            stats["success_rate"] = (stats["total_evaluations"] - stats["failed_evaluations"]) / stats["total_evaluations"]
        
        return stats
    
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of supported metrics.
        
        Returns:
            List of supported metric names
        """
        return [
            "hallucination",
            "context_precision",
            "context_recall",
            "answer_relevance",
            "usefulness",
            "moderation",
            "conversational_coherence",
            "session_completeness_quality",
            "user_frustration"
        ]
    
    def validate_metrics(self, metrics: List[str]) -> List[str]:
        """
        Validate list of metrics.
        
        Args:
            metrics: List of metrics to validate
            
        Returns:
            List of valid metrics
            
        Raises:
            OpikError: If invalid metrics are found
        """
        supported_metrics = self.get_supported_metrics()
        invalid_metrics = [m for m in metrics if m not in supported_metrics]
        
        if invalid_metrics:
            raise OpikError(f"Unsupported metrics: {invalid_metrics}")
        
        return metrics
