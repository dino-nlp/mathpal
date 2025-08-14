"""
Custom metrics evaluator for the evaluation pipeline.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..config import ConfigManager
from ..utils import get_logger
from .vietnamese_math_metrics import VietnameseMathMetrics, VietnameseMathMetricResult


@dataclass
class CustomMetricResult:
    """Result of custom metric evaluation."""
    
    metric_name: str
    score: float
    confidence: float
    explanation: str
    details: Dict[str, Any]


class CustomMetricsEvaluator:
    """
    Custom metrics evaluator for Vietnamese math evaluation.
    
    Combines Opik metrics with custom Vietnamese math-specific metrics.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize custom metrics evaluator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger("CustomMetricsEvaluator")
        
        # Initialize Vietnamese math metrics
        self.vietnamese_metrics = VietnameseMathMetrics()
        
        # Evaluation statistics
        self.evaluation_stats = {
            "total_evaluations": 0,
            "total_requests": 0,
            "total_time": 0.0,
            "failed_evaluations": 0
        }
        
        self.logger.info("Custom metrics evaluator initialized")
    
    def evaluate(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        expected_answers: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate using custom metrics.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            answers: List of answers
            expected_answers: Optional list of expected answers
            metadata: Optional list of metadata
            
        Returns:
            Dictionary of metric scores
            
        Raises:
            Exception: If evaluation fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting custom metrics evaluation of {len(questions)} samples")
            
            # Initialize results
            all_results = {}
            
            # Evaluate Vietnamese math metrics
            vietnamese_results = self._evaluate_vietnamese_metrics(
                questions, answers, expected_answers
            )
            all_results.update(vietnamese_results)
            
            # Evaluate additional custom metrics
            additional_results = self._evaluate_additional_metrics(
                questions, contexts, answers, expected_answers, metadata
            )
            all_results.update(additional_results)
            
            # Update statistics
            self._update_stats(start_time, len(questions))
            
            self.logger.info("Custom metrics evaluation completed successfully")
            return all_results
            
        except Exception as e:
            self.evaluation_stats["failed_evaluations"] += 1
            raise Exception(f"Custom metrics evaluation failed: {e}")
    
    def _evaluate_vietnamese_metrics(
        self,
        questions: List[str],
        answers: List[str],
        expected_answers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate Vietnamese math-specific metrics.
        
        Args:
            questions: List of questions
            answers: List of answers
            expected_answers: Optional list of expected answers
            
        Returns:
            Dictionary of Vietnamese math metric scores
        """
        results = {}
        
        # Initialize metric scores
        metric_scores = {
            "mathematical_accuracy": [],
            "vietnamese_language_quality": [],
            "step_by_step_reasoning": [],
            "grade_level_appropriateness": [],
            "problem_solving_approach": []
        }
        
        # Evaluate each sample
        for i, (question, answer) in enumerate(zip(questions, answers)):
            expected_answer = expected_answers[i] if expected_answers and i < len(expected_answers) else None
            
            try:
                # Mathematical accuracy
                math_result = self.vietnamese_metrics.evaluate_mathematical_accuracy(
                    question, answer, expected_answer
                )
                metric_scores["mathematical_accuracy"].append(math_result.score)
                
                # Vietnamese language quality
                lang_result = self.vietnamese_metrics.evaluate_vietnamese_language_quality(answer)
                metric_scores["vietnamese_language_quality"].append(lang_result.score)
                
                # Step-by-step reasoning
                reasoning_result = self.vietnamese_metrics.evaluate_step_by_step_reasoning(answer)
                metric_scores["step_by_step_reasoning"].append(reasoning_result.score)
                
                # Grade level appropriateness
                grade_result = self.vietnamese_metrics.evaluate_grade_level_appropriateness(
                    question, answer
                )
                metric_scores["grade_level_appropriateness"].append(grade_result.score)
                
                # Problem-solving approach
                approach_result = self.vietnamese_metrics.evaluate_problem_solving_approach(answer)
                metric_scores["problem_solving_approach"].append(approach_result.score)
                
            except Exception as e:
                self.logger.error(f"Error evaluating sample {i}: {e}")
                # Add default scores for failed evaluation
                for metric in metric_scores:
                    metric_scores[metric].append(0.0)
        
        # Calculate average scores
        for metric, scores in metric_scores.items():
            if scores:
                results[metric] = sum(scores) / len(scores)
            else:
                results[metric] = 0.0
        
        return results
    
    def _evaluate_additional_metrics(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        expected_answers: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate additional custom metrics.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            answers: List of answers
            expected_answers: Optional list of expected answers
            metadata: Optional list of metadata
            
        Returns:
            Dictionary of additional metric scores
        """
        results = {}
        
        # Answer completeness
        completeness_scores = []
        for answer in answers:
            score = self._evaluate_answer_completeness(answer)
            completeness_scores.append(score)
        
        if completeness_scores:
            results["answer_completeness"] = sum(completeness_scores) / len(completeness_scores)
        
        # Response consistency
        consistency_scores = []
        for i, answer in enumerate(answers):
            context = contexts[i] if i < len(contexts) else ""
            score = self._evaluate_response_consistency(answer, context)
            consistency_scores.append(score)
        
        if consistency_scores:
            results["response_consistency"] = sum(consistency_scores) / len(consistency_scores)
        
        # Educational value
        educational_scores = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            score = self._evaluate_educational_value(question, answer)
            educational_scores.append(score)
        
        if educational_scores:
            results["educational_value"] = sum(educational_scores) / len(educational_scores)
        
        # Clarity and readability
        clarity_scores = []
        for answer in answers:
            score = self._evaluate_clarity_readability(answer)
            clarity_scores.append(score)
        
        if clarity_scores:
            results["clarity_readability"] = sum(clarity_scores) / len(clarity_scores)
        
        return results
    
    def _evaluate_answer_completeness(self, answer: str) -> float:
        """
        Evaluate answer completeness.
        
        Args:
            answer: The model's answer
            
        Returns:
            Completeness score (0-1)
        """
        # Check for minimum content length
        word_count = len(answer.split())
        if word_count < 5:
            return 0.2
        elif word_count < 15:
            return 0.5
        elif word_count < 50:
            return 0.8
        else:
            return 1.0
    
    def _evaluate_response_consistency(self, answer: str, context: str) -> float:
        """
        Evaluate response consistency with context.
        
        Args:
            answer: The model's answer
            context: The provided context
            
        Returns:
            Consistency score (0-1)
        """
        if not context:
            return 0.8  # No context to check against
        
        # Check for context keywords in answer
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        if context_words and answer_words:
            overlap = len(context_words.intersection(answer_words))
            return min(1.0, overlap / max(len(context_words), 1))
        else:
            return 0.5
    
    def _evaluate_educational_value(self, question: str, answer: str) -> float:
        """
        Evaluate educational value of the answer.
        
        Args:
            question: The question
            answer: The model's answer
            
        Returns:
            Educational value score (0-1)
        """
        score = 0.5  # Base score
        
        # Check for educational elements
        educational_indicators = [
            "giải thích", "vì sao", "cách", "phương pháp", "công thức",
            "ví dụ", "minh họa", "bước", "đầu tiên", "tiếp theo"
        ]
        
        indicator_count = sum(1 for indicator in educational_indicators if indicator in answer.lower())
        score += min(0.4, indicator_count * 0.1)
        
        # Check for mathematical content
        math_symbols = ['+', '-', '*', '/', '=', '(', ')', 'x', 'y', 'z']
        math_count = sum(1 for symbol in math_symbols if symbol in answer)
        score += min(0.1, math_count * 0.02)
        
        return min(1.0, score)
    
    def _evaluate_clarity_readability(self, answer: str) -> float:
        """
        Evaluate clarity and readability of the answer.
        
        Args:
            answer: The model's answer
            
        Returns:
            Clarity score (0-1)
        """
        score = 0.5  # Base score
        
        # Check sentence structure
        sentences = answer.split('.')
        if len(sentences) > 1:
            score += 0.2
        
        # Check for proper punctuation
        if answer.endswith(('.', '!', '?')):
            score += 0.1
        
        # Check for Vietnamese characters
        vietnamese_chars = sum(1 for char in answer if ord(char) > 127)
        if vietnamese_chars > 10:
            score += 0.1
        
        # Check for logical connectors
        connectors = ['vì', 'do', 'nên', 'suy ra', 'từ đó', 'theo']
        connector_count = sum(1 for connector in connectors if connector in answer.lower())
        score += min(0.1, connector_count * 0.02)
        
        return min(1.0, score)
    
    def _update_stats(self, start_time: float, num_requests: int):
        """
        Update evaluation statistics.
        
        Args:
            start_time: Start time of evaluation
            num_requests: Number of requests processed
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
            # Vietnamese math metrics
            "mathematical_accuracy",
            "vietnamese_language_quality",
            "step_by_step_reasoning",
            "grade_level_appropriateness",
            "problem_solving_approach",
            
            # Additional custom metrics
            "answer_completeness",
            "response_consistency",
            "educational_value",
            "clarity_readability"
        ]
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all supported metrics.
        
        Returns:
            Dictionary of metric descriptions
        """
        return {
            "mathematical_accuracy": "Độ chính xác toán học của câu trả lời",
            "vietnamese_language_quality": "Chất lượng tiếng Việt trong câu trả lời",
            "step_by_step_reasoning": "Chất lượng lý luận từng bước",
            "grade_level_appropriateness": "Tính phù hợp với cấp độ học",
            "problem_solving_approach": "Phương pháp giải quyết vấn đề",
            "answer_completeness": "Tính đầy đủ của câu trả lời",
            "response_consistency": "Tính nhất quán với ngữ cảnh",
            "educational_value": "Giá trị giáo dục của câu trả lời",
            "clarity_readability": "Tính rõ ràng và dễ đọc"
        }
