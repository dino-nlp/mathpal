"""
Vietnamese math-specific evaluation metrics.
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..utils import get_logger


@dataclass
class VietnameseMathMetricResult:
    """Result of Vietnamese math metric evaluation."""
    
    metric_name: str
    score: float
    confidence: float
    explanation: str
    details: Dict[str, Any]


class VietnameseMathMetrics:
    """
    Vietnamese math-specific evaluation metrics.
    
    Provides specialized metrics for evaluating Vietnamese math education models.
    """
    
    def __init__(self):
        """Initialize Vietnamese math metrics."""
        self.logger = get_logger("VietnameseMathMetrics")
        
        # Vietnamese math patterns
        self.math_patterns = {
            "addition": r"(\d+)\s*\+\s*(\d+)",
            "subtraction": r"(\d+)\s*-\s*(\d+)",
            "multiplication": r"(\d+)\s*\*\s*(\d+)",
            "division": r"(\d+)\s*/\s*(\d+)",
            "equation": r"(\d+)x\s*[+\-]\s*(\d+)\s*=\s*(\d+)",
            "area": r"diện tích|diện tích hình|diện tích hình chữ nhật",
            "perimeter": r"chu vi|chu vi hình|chu vi hình chữ nhật",
            "fraction": r"(\d+)/(\d+)",
            "percentage": r"(\d+)%",
            "decimal": r"(\d+),(\d+)"
        }
        
        # Vietnamese language patterns
        self.vietnamese_patterns = {
            "math_terms": [
                "tính", "tìm", "giải", "tính toán", "kết quả", "đáp án",
                "phép tính", "phép cộng", "phép trừ", "phép nhân", "phép chia",
                "hình chữ nhật", "hình vuông", "hình tròn", "diện tích", "chu vi",
                "số", "số tự nhiên", "số thập phân", "phân số", "phần trăm"
            ],
            "step_indicators": [
                "bước", "đầu tiên", "tiếp theo", "sau đó", "cuối cùng",
                "trước tiên", "thứ nhất", "thứ hai", "thứ ba"
            ],
            "explanation_terms": [
                "vì", "do", "nên", "suy ra", "từ đó", "theo công thức",
                "áp dụng", "thay vào", "thế vào"
            ]
        }
        
        self.logger.info("Vietnamese math metrics initialized")
    
    def evaluate_mathematical_accuracy(
        self,
        question: str,
        answer: str,
        expected_answer: Optional[str] = None
    ) -> VietnameseMathMetricResult:
        """
        Evaluate mathematical accuracy of the answer.
        
        Args:
            question: The math question
            answer: The model's answer
            expected_answer: Expected answer (optional)
            
        Returns:
            Mathematical accuracy evaluation result
        """
        score = 0.0
        confidence = 0.0
        explanation = ""
        details = {}
        
        try:
            # Extract mathematical expressions from question and answer
            question_math = self._extract_math_expressions(question)
            answer_math = self._extract_math_expressions(answer)
            
            # Check if answer contains mathematical content
            if not answer_math:
                explanation = "Answer does not contain mathematical expressions"
                score = 0.1
                confidence = 0.8
            else:
                # Check mathematical correctness
                correctness_score = self._check_mathematical_correctness(
                    question_math, answer_math, expected_answer
                )
                
                # Check for step-by-step reasoning
                reasoning_score = self._check_step_by_step_reasoning(answer)
                
                # Check for proper mathematical notation
                notation_score = self._check_mathematical_notation(answer)
                
                # Calculate overall score
                score = (correctness_score * 0.6 + reasoning_score * 0.3 + notation_score * 0.1)
                
                explanation = f"Mathematical accuracy: correctness={correctness_score:.2f}, reasoning={reasoning_score:.2f}, notation={notation_score:.2f}"
                confidence = 0.9
                
                details = {
                    "correctness_score": correctness_score,
                    "reasoning_score": reasoning_score,
                    "notation_score": notation_score,
                    "question_math": question_math,
                    "answer_math": answer_math
                }
        
        except Exception as e:
            self.logger.error(f"Error evaluating mathematical accuracy: {e}")
            explanation = f"Error in evaluation: {str(e)}"
            score = 0.0
            confidence = 0.5
        
        return VietnameseMathMetricResult(
            metric_name="mathematical_accuracy",
            score=score,
            confidence=confidence,
            explanation=explanation,
            details=details
        )
    
    def evaluate_vietnamese_language_quality(self, answer: str) -> VietnameseMathMetricResult:
        """
        Evaluate Vietnamese language quality in the answer.
        
        Args:
            answer: The model's answer
            
        Returns:
            Vietnamese language quality evaluation result
        """
        score = 0.0
        confidence = 0.0
        explanation = ""
        details = {}
        
        try:
            # Check for Vietnamese characters
            vietnamese_chars = self._count_vietnamese_characters(answer)
            char_score = min(1.0, vietnamese_chars / 10.0)  # Normalize to 0-1
            
            # Check for proper Vietnamese math terminology
            terminology_score = self._check_vietnamese_terminology(answer)
            
            # Check for grammatical correctness
            grammar_score = self._check_vietnamese_grammar(answer)
            
            # Check for sentence structure
            structure_score = self._check_sentence_structure(answer)
            
            # Calculate overall score
            score = (char_score * 0.2 + terminology_score * 0.4 + grammar_score * 0.3 + structure_score * 0.1)
            
            explanation = f"Vietnamese quality: chars={char_score:.2f}, terminology={terminology_score:.2f}, grammar={grammar_score:.2f}, structure={structure_score:.2f}"
            confidence = 0.8
            
            details = {
                "vietnamese_chars": vietnamese_chars,
                "char_score": char_score,
                "terminology_score": terminology_score,
                "grammar_score": grammar_score,
                "structure_score": structure_score
            }
        
        except Exception as e:
            self.logger.error(f"Error evaluating Vietnamese language quality: {e}")
            explanation = f"Error in evaluation: {str(e)}"
            score = 0.0
            confidence = 0.5
        
        return VietnameseMathMetricResult(
            metric_name="vietnamese_language_quality",
            score=score,
            confidence=confidence,
            explanation=explanation,
            details=details
        )
    
    def evaluate_step_by_step_reasoning(self, answer: str) -> VietnameseMathMetricResult:
        """
        Evaluate step-by-step reasoning quality.
        
        Args:
            answer: The model's answer
            
        Returns:
            Step-by-step reasoning evaluation result
        """
        score = 0.0
        confidence = 0.0
        explanation = ""
        details = {}
        
        try:
            # Check for step indicators
            step_indicators = self._count_step_indicators(answer)
            step_score = min(1.0, step_indicators / 3.0)  # Normalize to 0-1
            
            # Check for logical flow
            flow_score = self._check_logical_flow(answer)
            
            # Check for mathematical expressions in steps
            math_in_steps = self._check_math_in_steps(answer)
            
            # Check for explanation quality
            explanation_score = self._check_explanation_quality(answer)
            
            # Calculate overall score
            score = (step_score * 0.3 + flow_score * 0.3 + math_in_steps * 0.2 + explanation_score * 0.2)
            
            explanation = f"Step-by-step reasoning: steps={step_score:.2f}, flow={flow_score:.2f}, math={math_in_steps:.2f}, explanation={explanation_score:.2f}"
            confidence = 0.85
            
            details = {
                "step_indicators": step_indicators,
                "step_score": step_score,
                "flow_score": flow_score,
                "math_in_steps": math_in_steps,
                "explanation_score": explanation_score
            }
        
        except Exception as e:
            self.logger.error(f"Error evaluating step-by-step reasoning: {e}")
            explanation = f"Error in evaluation: {str(e)}"
            score = 0.0
            confidence = 0.5
        
        return VietnameseMathMetricResult(
            metric_name="step_by_step_reasoning",
            score=score,
            confidence=confidence,
            explanation=explanation,
            details=details
        )
    
    def evaluate_grade_level_appropriateness(
        self,
        question: str,
        answer: str,
        target_grade: Optional[str] = None
    ) -> VietnameseMathMetricResult:
        """
        Evaluate if the answer is appropriate for the target grade level.
        
        Args:
            question: The math question
            answer: The model's answer
            target_grade: Target grade level (optional)
            
        Returns:
            Grade level appropriateness evaluation result
        """
        score = 0.0
        confidence = 0.0
        explanation = ""
        details = {}
        
        try:
            # Determine expected grade level from question
            question_grade = self._determine_question_grade_level(question)
            
            # Determine answer complexity
            answer_complexity = self._determine_answer_complexity(answer)
            
            # Check vocabulary appropriateness
            vocab_score = self._check_vocabulary_appropriateness(answer, question_grade)
            
            # Check mathematical concept appropriateness
            concept_score = self._check_concept_appropriateness(answer, question_grade)
            
            # Check explanation complexity
            explanation_complexity = self._check_explanation_complexity(answer, question_grade)
            
            # Calculate overall score
            score = (vocab_score * 0.4 + concept_score * 0.4 + explanation_complexity * 0.2)
            
            explanation = f"Grade appropriateness: vocab={vocab_score:.2f}, concept={concept_score:.2f}, explanation={explanation_complexity:.2f}"
            confidence = 0.8
            
            details = {
                "question_grade": question_grade,
                "answer_complexity": answer_complexity,
                "vocab_score": vocab_score,
                "concept_score": concept_score,
                "explanation_complexity": explanation_complexity
            }
        
        except Exception as e:
            self.logger.error(f"Error evaluating grade level appropriateness: {e}")
            explanation = f"Error in evaluation: {str(e)}"
            score = 0.0
            confidence = 0.5
        
        return VietnameseMathMetricResult(
            metric_name="grade_level_appropriateness",
            score=score,
            confidence=confidence,
            explanation=explanation,
            details=details
        )
    
    def evaluate_problem_solving_approach(self, answer: str) -> VietnameseMathMetricResult:
        """
        Evaluate problem-solving approach quality.
        
        Args:
            answer: The model's answer
            
        Returns:
            Problem-solving approach evaluation result
        """
        score = 0.0
        confidence = 0.0
        explanation = ""
        details = {}
        
        try:
            # Check for systematic approach
            systematic_score = self._check_systematic_approach(answer)
            
            # Check for problem identification
            identification_score = self._check_problem_identification(answer)
            
            # Check for solution strategy
            strategy_score = self._check_solution_strategy(answer)
            
            # Check for verification
            verification_score = self._check_verification(answer)
            
            # Calculate overall score
            score = (systematic_score * 0.3 + identification_score * 0.2 + strategy_score * 0.3 + verification_score * 0.2)
            
            explanation = f"Problem-solving approach: systematic={systematic_score:.2f}, identification={identification_score:.2f}, strategy={strategy_score:.2f}, verification={verification_score:.2f}"
            confidence = 0.85
            
            details = {
                "systematic_score": systematic_score,
                "identification_score": identification_score,
                "strategy_score": strategy_score,
                "verification_score": verification_score
            }
        
        except Exception as e:
            self.logger.error(f"Error evaluating problem-solving approach: {e}")
            explanation = f"Error in evaluation: {str(e)}"
            score = 0.0
            confidence = 0.5
        
        return VietnameseMathMetricResult(
            metric_name="problem_solving_approach",
            score=score,
            confidence=confidence,
            explanation=explanation,
            details=details
        )
    
    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        expressions = []
        
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            expressions.extend(matches)
        
        return expressions
    
    def _check_mathematical_correctness(
        self,
        question_math: List[str],
        answer_math: List[str],
        expected_answer: Optional[str]
    ) -> float:
        """Check mathematical correctness."""
        if expected_answer:
            # Compare with expected answer
            expected_math = self._extract_math_expressions(expected_answer)
            if answer_math and expected_math:
                # Simple comparison - in real implementation, would use more sophisticated math evaluation
                return 0.8 if len(answer_math) > 0 else 0.2
        
        # Check if answer contains mathematical content
        if answer_math:
            return 0.7
        else:
            return 0.2
    
    def _check_step_by_step_reasoning(self, answer: str) -> float:
        """Check step-by-step reasoning quality."""
        step_indicators = self._count_step_indicators(answer)
        math_expressions = len(self._extract_math_expressions(answer))
        
        if step_indicators > 0 and math_expressions > 0:
            return 0.8
        elif step_indicators > 0:
            return 0.6
        elif math_expressions > 0:
            return 0.4
        else:
            return 0.2
    
    def _check_mathematical_notation(self, answer: str) -> float:
        """Check mathematical notation quality."""
        # Check for proper mathematical symbols
        symbols = ['+', '-', '*', '/', '=', '(', ')', 'x', 'y', 'z']
        symbol_count = sum(1 for symbol in symbols if symbol in answer)
        
        return min(1.0, symbol_count / 5.0)
    
    def _count_vietnamese_characters(self, text: str) -> int:
        """Count Vietnamese characters in text."""
        vietnamese_chars = 0
        for char in text:
            if ord(char) > 127:  # Non-ASCII characters
                vietnamese_chars += 1
        return vietnamese_chars
    
    def _check_vietnamese_terminology(self, answer: str) -> float:
        """Check Vietnamese math terminology usage."""
        terminology_count = 0
        for term in self.vietnamese_patterns["math_terms"]:
            if term.lower() in answer.lower():
                terminology_count += 1
        
        return min(1.0, terminology_count / 5.0)
    
    def _check_vietnamese_grammar(self, answer: str) -> float:
        """Check Vietnamese grammar quality."""
        # Simple heuristics for Vietnamese grammar
        sentences = answer.split('.')
        if len(sentences) > 1:
            return 0.8
        elif len(answer.split()) > 5:
            return 0.6
        else:
            return 0.3
    
    def _check_sentence_structure(self, answer: str) -> float:
        """Check sentence structure quality."""
        # Check for proper sentence endings
        if answer.endswith(('.', '!', '?')):
            return 0.8
        else:
            return 0.5
    
    def _count_step_indicators(self, answer: str) -> int:
        """Count step indicators in answer."""
        count = 0
        for indicator in self.vietnamese_patterns["step_indicators"]:
            if indicator.lower() in answer.lower():
                count += 1
        return count
    
    def _check_logical_flow(self, answer: str) -> float:
        """Check logical flow of reasoning."""
        # Check for logical connectors
        connectors = ['vì', 'do', 'nên', 'suy ra', 'từ đó', 'theo']
        connector_count = sum(1 for connector in connectors if connector in answer.lower())
        
        return min(1.0, connector_count / 3.0)
    
    def _check_math_in_steps(self, answer: str) -> float:
        """Check for mathematical expressions in reasoning steps."""
        math_expressions = self._extract_math_expressions(answer)
        return min(1.0, len(math_expressions) / 3.0)
    
    def _check_explanation_quality(self, answer: str) -> float:
        """Check explanation quality."""
        explanation_terms = self.vietnamese_patterns["explanation_terms"]
        term_count = sum(1 for term in explanation_terms if term in answer.lower())
        
        return min(1.0, term_count / 2.0)
    
    def _determine_question_grade_level(self, question: str) -> str:
        """Determine expected grade level from question."""
        # Simple heuristics based on mathematical concepts
        if any(word in question.lower() for word in ['phương trình', 'bất phương trình', 'hàm số']):
            return "8"
        elif any(word in question.lower() for word in ['phân số', 'số thập phân', 'phần trăm']):
            return "6"
        elif any(word in question.lower() for word in ['phép nhân', 'phép chia', 'hình học']):
            return "5"
        else:
            return "5"  # Default to grade 5
    
    def _determine_answer_complexity(self, answer: str) -> str:
        """Determine complexity level of answer."""
        word_count = len(answer.split())
        if word_count > 50:
            return "high"
        elif word_count > 20:
            return "medium"
        else:
            return "low"
    
    def _check_vocabulary_appropriateness(self, answer: str, grade_level: str) -> float:
        """Check vocabulary appropriateness for grade level."""
        # Simple heuristics - in real implementation, would use vocabulary lists
        complex_words = ['phương trình', 'bất phương trình', 'hàm số', 'đạo hàm', 'tích phân']
        complex_count = sum(1 for word in complex_words if word in answer.lower())
        
        if grade_level in ["5", "6"] and complex_count > 0:
            return 0.3
        elif grade_level in ["7", "8"] and complex_count > 2:
            return 0.3
        else:
            return 0.8
    
    def _check_concept_appropriateness(self, answer: str, grade_level: str) -> float:
        """Check mathematical concept appropriateness for grade level."""
        # Simple heuristics
        advanced_concepts = ['phương trình', 'bất phương trình', 'hàm số']
        concept_count = sum(1 for concept in advanced_concepts if concept in answer.lower())
        
        if grade_level in ["5", "6"] and concept_count > 0:
            return 0.4
        else:
            return 0.8
    
    def _check_explanation_complexity(self, answer: str, grade_level: str) -> float:
        """Check explanation complexity for grade level."""
        word_count = len(answer.split())
        
        if grade_level in ["5", "6"] and word_count > 100:
            return 0.4
        elif grade_level in ["7", "8"] and word_count > 200:
            return 0.4
        else:
            return 0.8
    
    def _check_systematic_approach(self, answer: str) -> float:
        """Check for systematic problem-solving approach."""
        systematic_indicators = ['đầu tiên', 'tiếp theo', 'sau đó', 'cuối cùng', 'bước']
        indicator_count = sum(1 for indicator in systematic_indicators if indicator in answer.lower())
        
        return min(1.0, indicator_count / 3.0)
    
    def _check_problem_identification(self, answer: str) -> float:
        """Check for problem identification."""
        identification_indicators = ['bài toán', 'vấn đề', 'cần tìm', 'yêu cầu']
        indicator_count = sum(1 for indicator in identification_indicators if indicator in answer.lower())
        
        return min(1.0, indicator_count / 2.0)
    
    def _check_solution_strategy(self, answer: str) -> float:
        """Check for solution strategy."""
        strategy_indicators = ['cách giải', 'phương pháp', 'công thức', 'áp dụng']
        indicator_count = sum(1 for indicator in strategy_indicators if indicator in answer.lower())
        
        return min(1.0, indicator_count / 2.0)
    
    def _check_verification(self, answer: str) -> float:
        """Check for answer verification."""
        verification_indicators = ['kiểm tra', 'thử lại', 'xác nhận', 'đúng']
        indicator_count = sum(1 for indicator in verification_indicators if indicator in answer.lower())
        
        return min(1.0, indicator_count / 2.0)
