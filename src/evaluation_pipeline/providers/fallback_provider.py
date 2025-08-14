"""
Fallback provider for LLM-as-a-judge evaluation.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..utils import get_logger


@dataclass
class FallbackResponse:
    """Response from fallback provider."""
    content: str
    model: str = "fallback"
    cost: Optional[float] = None


class FallbackProvider:
    """Fallback provider for LLM-as-a-judge evaluation."""
    
    def __init__(self):
        """Initialize fallback provider."""
        self.logger = get_logger("FallbackProvider")
        self.logger.info("Fallback provider initialized")
    
    def evaluate_as_judge(
        self,
        question: str,
        context: str,
        answer: str,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use fallback logic to evaluate an answer."""
        self.logger.info("Using fallback evaluation logic")
        
        # Simple heuristic-based evaluation
        scores = {}
        
        # Accuracy: Check if answer contains mathematical content
        if any(char in answer for char in ['+', '-', '*', '/', '=', '(', ')', 'x', 'y', 'z']):
            scores["accuracy"] = {"score": 7.0, "explanation": "Contains mathematical expressions"}
        else:
            scores["accuracy"] = {"score": 4.0, "explanation": "No mathematical expressions found"}
        
        # Completeness: Check answer length
        word_count = len(answer.split())
        if word_count > 20:
            scores["completeness"] = {"score": 8.0, "explanation": "Detailed answer"}
        elif word_count > 10:
            scores["completeness"] = {"score": 6.0, "explanation": "Moderate detail"}
        else:
            scores["completeness"] = {"score": 3.0, "explanation": "Brief answer"}
        
        # Clarity: Check for Vietnamese characters
        vietnamese_chars = sum(1 for char in answer if ord(char) > 127)
        if vietnamese_chars > 10:
            scores["clarity"] = {"score": 8.0, "explanation": "Good Vietnamese language use"}
        elif vietnamese_chars > 5:
            scores["clarity"] = {"score": 6.0, "explanation": "Some Vietnamese content"}
        else:
            scores["clarity"] = {"score": 4.0, "explanation": "Limited Vietnamese content"}
        
        # Relevance: Check if answer mentions question keywords
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words.intersection(answer_words))
        if overlap > 2:
            scores["relevance"] = {"score": 8.0, "explanation": "Good keyword overlap"}
        elif overlap > 0:
            scores["relevance"] = {"score": 6.0, "explanation": "Some keyword overlap"}
        else:
            scores["relevance"] = {"score": 4.0, "explanation": "No keyword overlap"}
        
        # Helpfulness: Check for step indicators
        step_indicators = ['bước', 'đầu tiên', 'tiếp theo', 'sau đó', 'cuối cùng']
        step_count = sum(1 for indicator in step_indicators if indicator in answer.lower())
        if step_count > 1:
            scores["helpfulness"] = {"score": 8.0, "explanation": "Step-by-step explanation"}
        elif step_count > 0:
            scores["helpfulness"] = {"score": 6.0, "explanation": "Some step indicators"}
        else:
            scores["helpfulness"] = {"score": 4.0, "explanation": "No step indicators"}
        
        # Calculate overall score
        overall_score = sum(s["score"] for s in scores.values()) / len(scores)
        
        return {
            "scores": scores,
            "overall_score": overall_score,
            "final_assessment": "Fallback evaluation completed using heuristic rules",
            "model_used": "fallback",
            "cost": 0.0
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": 0,
            "total_cost": 0.0,
            "provider": "fallback"
        }
