"""
Custom metrics with progress tracking for Opik evaluation.
"""

import time
from typing import Dict, Any, Optional
from tqdm import tqdm
from opik.evaluation.metrics import base_metric, score_result
from core.logger_utils import get_logger

logger = get_logger(__name__)


class ProgressTrackingMetric(base_metric.BaseMetric):
    """
    Base class for metrics that support progress tracking.
    """
    
    def __init__(self, name: str, track_progress: bool = True):
        super().__init__(name)
        self.track_progress = track_progress
        self.progress_bar = None
        self.total_calls = 0
        self.current_call = 0
        
    def _init_progress_bar(self, total: int, desc: str):
        """Initialize progress bar for this metric"""
        if self.track_progress:
            self.total_calls = total
            self.progress_bar = tqdm(
                total=total,
                desc=f"📊 {desc}",
                unit="call",
                leave=False,  # Don't leave the bar after completion
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
    
    def _update_progress(self, additional_info: Optional[Dict[str, Any]] = None):
        """Update progress bar"""
        if self.progress_bar and self.track_progress:
            self.current_call += 1
            postfix = additional_info or {}
            self.progress_bar.set_postfix(postfix)
            self.progress_bar.update(1)
    
    def _close_progress_bar(self):
        """Close progress bar"""
        if self.progress_bar and self.track_progress:
            self.progress_bar.close()
            self.progress_bar = None


class ProgressLevenshteinRatio(ProgressTrackingMetric):
    """
    Levenshtein ratio metric with progress tracking.
    """
    
    def __init__(self, name: str = "progress_levenshtein_ratio", track_progress: bool = True):
        super().__init__(name, track_progress)
        
    def score(self, output: str, reference: str, **kwargs) -> score_result.ScoreResult:
        """Calculate Levenshtein ratio with progress tracking"""
        try:
            # Simple Levenshtein distance calculation
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(output, reference)
            max_len = max(len(output), len(reference))
            ratio = 1 - (distance / max_len) if max_len > 0 else 1.0
            
            # Update progress
            self._update_progress({
                "Ratio": f"{ratio:.3f}",
                "Distance": distance
            })
            
            return score_result.ScoreResult(
                name=self.name,
                value=ratio,
                reason=f"Levenshtein ratio: {ratio:.3f} (distance: {distance}, max_len: {max_len})"
            )
            
        except Exception as e:
            logger.error(f"Error calculating Levenshtein ratio: {e}")
            return score_result.ScoreResult(
                name=self.name,
                value=0.0,
                reason=f"Error: {str(e)}"
            )


class ProgressHallucination(ProgressTrackingMetric):
    """
    Hallucination detection metric with progress tracking.
    """
    
    def __init__(self, name: str = "progress_hallucination", track_progress: bool = True):
        super().__init__(name, track_progress)
        
    def score(self, input: str, output: str, **kwargs) -> score_result.ScoreResult:
        """Detect hallucination with progress tracking"""
        try:
            # Simple heuristic-based hallucination detection
            # This is a simplified version - in practice, you'd use a more sophisticated approach
            
            # Check for common hallucination indicators
            hallucination_indicators = [
                "I don't have enough information",
                "I cannot provide",
                "I'm not sure",
                "I don't know",
                "I cannot answer",
                "I don't have access to",
                "I'm unable to",
                "I cannot determine"
            ]
            
            # Check if output contains hallucination indicators
            has_indicators = any(indicator.lower() in output.lower() for indicator in hallucination_indicators)
            
            # Check if output is too short (potential incomplete response)
            is_too_short = len(output.strip()) < 10
            
            # Check if output repeats input (potential copying without understanding)
            input_words = set(input.lower().split())
            output_words = set(output.lower().split())
            overlap_ratio = len(input_words.intersection(output_words)) / len(input_words) if input_words else 0
            is_repetitive = overlap_ratio > 0.8
            
            # Calculate hallucination score
            hallucination_score = 0.0
            reasons = []
            
            if has_indicators:
                hallucination_score += 0.3
                reasons.append("Contains uncertainty indicators")
            
            if is_too_short:
                hallucination_score += 0.2
                reasons.append("Response too short")
                
            if is_repetitive:
                hallucination_score += 0.5
                reasons.append("High input repetition")
            
            # Normalize score to 0-1 range
            hallucination_score = min(hallucination_score, 1.0)
            
            # Update progress
            self._update_progress({
                "Score": f"{hallucination_score:.3f}",
                "Indicators": len([i for i in hallucination_indicators if i.lower() in output.lower()])
            })
            
            return score_result.ScoreResult(
                name=self.name,
                value=hallucination_score,
                reason=f"Hallucination score: {hallucination_score:.3f} - {'; '.join(reasons) if reasons else 'No clear indicators'}"
            )
            
        except Exception as e:
            logger.error(f"Error detecting hallucination: {e}")
            return score_result.ScoreResult(
                name=self.name,
                value=0.5,  # Neutral score on error
                reason=f"Error: {str(e)}"
            )


class ProgressModeration(ProgressTrackingMetric):
    """
    Content moderation metric with progress tracking.
    """
    
    def __init__(self, name: str = "progress_moderation", track_progress: bool = True):
        super().__init__(name, track_progress)
        
    def score(self, output: str, **kwargs) -> score_result.ScoreResult:
        """Moderate content with progress tracking"""
        try:
            # Simple content moderation check
            # This is a simplified version - in practice, you'd use a more sophisticated approach
            
            # Define inappropriate content patterns
            inappropriate_patterns = [
                "hate speech",
                "violence",
                "discrimination",
                "harassment",
                "inappropriate",
                "offensive"
            ]
            
            # Check for inappropriate content
            inappropriate_count = sum(1 for pattern in inappropriate_patterns 
                                   if pattern.lower() in output.lower())
            
            # Calculate moderation score (0 = appropriate, 1 = inappropriate)
            moderation_score = min(inappropriate_count * 0.2, 1.0)
            
            # Update progress
            self._update_progress({
                "Score": f"{moderation_score:.3f}",
                "Flags": inappropriate_count
            })
            
            return score_result.ScoreResult(
                name=self.name,
                value=moderation_score,
                reason=f"Moderation score: {moderation_score:.3f} - {inappropriate_count} potential flags"
            )
            
        except Exception as e:
            logger.error(f"Error in content moderation: {e}")
            return score_result.ScoreResult(
                name=self.name,
                value=0.0,  # Assume appropriate on error
                reason=f"Error: {str(e)}"
            )


class ProgressStyle(ProgressTrackingMetric):
    """
    Style evaluation metric with progress tracking.
    """
    
    def __init__(self, name: str = "progress_style", track_progress: bool = True):
        super().__init__(name, track_progress)
        
    def score(self, output: str, problem: str = None, **kwargs) -> score_result.ScoreResult:
        """Evaluate style with progress tracking"""
        try:
            # Simple style evaluation
            # This is a simplified version - in practice, you'd use a more sophisticated approach
            
            # Check for mathematical notation and formatting
            math_indicators = [
                "=", "+", "-", "*", "/", "(", ")", "[", "]", "{", "}",
                "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
            ]
            
            # Check for clear explanations
            explanation_indicators = [
                "because", "therefore", "thus", "hence", "so", "since", "as", "for",
                "step", "steps", "solution", "answer", "result", "calculate", "compute"
            ]
            
            # Count math indicators
            math_count = sum(1 for indicator in math_indicators if indicator in output)
            
            # Count explanation indicators
            explanation_count = sum(1 for indicator in explanation_indicators if indicator.lower() in output.lower())
            
            # Calculate style score based on mathematical content and explanations
            math_score = min(math_count / 10.0, 1.0)  # Normalize math content
            explanation_score = min(explanation_count / 5.0, 1.0)  # Normalize explanations
            
            # Combined style score
            style_score = (math_score + explanation_score) / 2.0
            
            # Update progress
            self._update_progress({
                "Score": f"{style_score:.3f}",
                "Math": math_count,
                "Explanations": explanation_count
            })
            
            return score_result.ScoreResult(
                name=self.name,
                value=style_score,
                reason=f"Style score: {style_score:.3f} - Math indicators: {math_count}, Explanation indicators: {explanation_count}"
            )
            
        except Exception as e:
            logger.error(f"Error in style evaluation: {e}")
            return score_result.ScoreResult(
                name=self.name,
                value=0.5,  # Neutral score on error
                reason=f"Error: {str(e)}"
            )


def create_progress_metrics(track_progress: bool = True) -> list:
    """
    Create a list of progress-tracking metrics.
    
    Args:
        track_progress: Whether to enable progress tracking
        
    Returns:
        List of metric instances
    """
    return [
        ProgressLevenshteinRatio(track_progress=track_progress),
        ProgressHallucination(track_progress=track_progress),
        ProgressModeration(track_progress=track_progress),
        ProgressStyle(track_progress=track_progress),
    ]


def setup_metric_progress_bars(metrics: list, total_samples: int):
    """
    Setup progress bars for all metrics.
    
    Args:
        metrics: List of ProgressTrackingMetric instances
        total_samples: Total number of samples to process
    """
    for metric in metrics:
        if isinstance(metric, ProgressTrackingMetric):
            metric._init_progress_bar(total_samples, metric.name)


def cleanup_metric_progress_bars(metrics: list):
    """
    Cleanup progress bars for all metrics.
    
    Args:
        metrics: List of ProgressTrackingMetric instances
    """
    for metric in metrics:
        if isinstance(metric, ProgressTrackingMetric):
            metric._close_progress_bar()
