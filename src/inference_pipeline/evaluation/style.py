import json
from typing import Any

from config import settings
from opik.evaluation.metrics import base_metric, score_result
from opik.evaluation.models import LiteLLMChatModel
from opik.exceptions import MetricComputationError
from pydantic import BaseModel


class LLMJudgeStyleOutputResult(BaseModel):
    score: int
    reason: str


class Style(base_metric.BaseMetric):
    """
    A metric that evaluates whether an LLM's output tone and writing style are appropriate for a blog post or social media content.

    This metric uses another LLM to judge if the output is factual or contains hallucinations.
    It returns a score of 1.0 if the style is appropriate, 0.5 if it is somewhere in the middle and 0.0 otherwise.
    """

    def __init__(
        self, name: str = "style_metric", model_name: str = settings.OPENROUTER_BASE_MODEL
    ) -> None:
        self.name = name
        self.llm_client = LiteLLMChatModel(model_name=model_name)
        self.prompt_template = """
        You are an impartial expert judge specializing in mathematics and educational pedagogy.
Evaluate the quality of a given solution to a math problem based on two criteria: Accuracy and Clarity & Pedagogical Value.

## Evaluation Criteria

### 1. Accuracy
Is the final answer correct? Are the steps, reasoning, and formulas used accurate and logical?

**Accuracy Scale:**
* **1 (Poor):** The solution is completely incorrect or contains critical mathematical errors (e.g., wrong formula, flawed logical reasoning).
* **2 (Good):** The final result is correct, but there are minor non-critical errors (e.g., typos, a less-than-rigorous step). Alternatively, the approach is correct, but there's a calculation mistake in the final step.
* **3 (Excellent):** The solution is entirely correct in its reasoning, formulas, calculations, and final result.

### 2. Clarity & Pedagogical Value
Is the solution presented in a logical, sequential, and easy-to-understand manner for a student? Are the steps adequately explained? Is the language appropriate for a learning context?

**Clarity & Pedagogical Value Scale:**
* **1 (Poor):** The solution is hard to follow, disorganized, or presents calculations without explanation. It uses overly complex or inappropriate terminology.
* **2 (Good):** The solution is structured and lists the main steps, but the explanation is brief and doesn't clarify the "why" behind each step.
* **3 (Excellent):** The presentation is exceptionally clear and logical, with each step explained thoroughly. It not only shows "how" to solve the problem but also helps the student understand "why" the method works. The language is natural and easy to comprehend.

---

Problem: {problem}

Solution: {solution}

---

Provide your evaluation in JSON format with the following structure:
{
    "reason": "...",
    "score": 1
}
"""

    def score(self, problem: str, solution: str, **ignored_kwargs: Any):
        """
        Score the output of an LLM.

        Args:
            output: The output of an LLM to score.
            **ignored_kwargs: Any additional keyword arguments. This is important so that the metric can be used in the `evaluate` function.
        """

        prompt = self.prompt_template.format(problem=problem, solution=solution)

        model_output = self.llm_client.generate_string(
            input=prompt, response_format=LLMJudgeStyleOutputResult
        )

        return self._parse_model_output(model_output)

    def _parse_model_output(self, content: str) -> score_result.ScoreResult:
        try:
            dict_content = json.loads(content)
        except Exception:
            raise MetricComputationError("Failed to parse the model output.")

        score = dict_content["score"]
        try:
            assert 1 <= score <= 3, f"Invalid score value: {score}"
        except AssertionError as e:
            raise MetricComputationError(str(e))

        score = (score - 1) / 2.0  # Normalize the score to be between 0 and 1

        return score_result.ScoreResult(
            name=self.name,
            value=score,
            reason=dict_content["reason"],
        )
