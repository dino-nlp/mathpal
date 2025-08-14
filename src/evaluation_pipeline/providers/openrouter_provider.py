"""
OpenRouter provider for LLM-as-a-judge evaluation.
"""

import time
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..config import ConfigManager
from ..utils import ProviderError, get_logger


@dataclass
class OpenRouterResponse:
    """Response from OpenRouter API."""
    content: str
    model: str = ""
    cost: Optional[float] = None


class OpenRouterProvider:
    """OpenRouter provider for LLM-as-a-judge evaluation."""
    
    def __init__(self, config: ConfigManager):
        """Initialize OpenRouter provider."""
        self.config = config
        self.openrouter_config = config.get_openrouter_config()
        self.logger = get_logger("OpenRouterProvider")
        
        # Read API key from environment variable for security
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY environment variable not set")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = self.openrouter_config.models.get("primary", "anthropic/claude-3.5-sonnet")
        
        self.request_count = 0
        self.total_cost = 0.0
        self.rate_limit = self.openrouter_config.rate_limits
        
        # Debug: Check API key
        self.logger.info(f"OpenRouter API Key: {self.api_key[:10]}...{self.api_key[-10:] if len(self.api_key) > 20 else '***'}")
        self.logger.info(f"OpenRouter Model: {self.default_model}")
        self.logger.info("OpenRouter provider initialized")
    
    def evaluate_as_judge(
        self,
        question: str,
        context: str,
        answer: str,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use LLM-as-a-judge to evaluate an answer."""
        try:
            prompt = self._create_evaluation_prompt(
                question, context, answer, expected_answer
            )
            
            response = self._make_chat_request(prompt)
            results = self._parse_evaluation_response(response.content)
            
            results.update({
                "model_used": response.model,
                "cost": response.cost
            })
            
            return results
            
        except Exception as e:
            raise ProviderError(f"LLM-as-a-judge evaluation failed: {e}")
    
    def _create_evaluation_prompt(
        self,
        question: str,
        context: str,
        answer: str,
        expected_answer: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Create evaluation prompt for LLM-as-a-judge."""
        system_prompt = """You are an expert evaluator for Vietnamese math education. Evaluate the quality of an AI model's answer to a math question.

Evaluation Criteria:
- Accuracy: Rate from 1-10
- Completeness: Rate from 1-10  
- Clarity: Rate from 1-10
- Relevance: Rate from 1-10
- Helpfulness: Rate from 1-10

Respond in JSON format:
{
    "scores": {
        "accuracy": {"score": X, "explanation": "..."},
        "completeness": {"score": X, "explanation": "..."},
        "clarity": {"score": X, "explanation": "..."},
        "relevance": {"score": X, "explanation": "..."},
        "helpfulness": {"score": X, "explanation": "..."}
    },
    "overall_score": X.X,
    "final_assessment": "..."
}"""

        user_prompt = f"""Question: {question}
Context: {context}
AI Answer: {answer}"""

        if expected_answer:
            user_prompt += f"\nExpected Answer: {expected_answer}"
        
        user_prompt += "\n\nPlease evaluate this answer."
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _make_chat_request(self, messages: List[Dict[str, str]]) -> OpenRouterResponse:
        """Make chat completion request to OpenRouter."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mathpal.ai",
                "X-Title": "MathPal Evaluation"
            }
            
            payload = {
                "model": self.default_model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return OpenRouterResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.default_model)
            )
            
        except Exception as e:
            raise ProviderError(f"Chat request failed: {e}")
    
    def _parse_evaluation_response(self, content: str) -> Dict[str, Any]:
        """Parse evaluation response from LLM."""
        try:
            # Debug: Log the raw response
            self.logger.debug(f"Raw OpenRouter response: {content[:500]}...")
            
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                self.logger.debug(f"Extracted JSON: {json_str}")
                parsed = json.loads(json_str)
                
                scores = {}
                overall_score = 0.0
                
                if "scores" in parsed:
                    for criterion, score_data in parsed["scores"].items():
                        if isinstance(score_data, dict):
                            scores[criterion] = {
                                "score": float(score_data.get("score", 0)),
                                "explanation": score_data.get("explanation", "")
                            }
                        else:
                            scores[criterion] = {
                                "score": float(score_data),
                                "explanation": ""
                            }
                
                if scores:
                    overall_score = sum(s["score"] for s in scores.values()) / len(scores)
                
                return {
                    "scores": scores,
                    "overall_score": overall_score,
                    "final_assessment": parsed.get("final_assessment", ""),
                    "raw_response": content
                }
            
            return {
                "scores": {},
                "overall_score": 5.0,
                "final_assessment": "Unable to parse evaluation",
                "raw_response": content
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing evaluation response: {e}")
            return {
                "scores": {},
                "overall_score": 5.0,
                "final_assessment": f"Parsing error: {e}",
                "raw_response": content
            }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.request_count,
            "total_cost": self.total_cost,
            "rate_limit": self.rate_limit
        }
