import os
import pprint
import torch
import opik
from opik import opik_context
from typing import Dict, List, Any, Optional
from inference_pipeline.config import settings
from core import logger_utils
from core.opik_utils import add_to_dataset_with_sampling
from inference_pipeline.utils import compute_num_tokens, truncate_text_to_max_tokens

from unsloth import FastModel, get_chat_template

logger = logger_utils.get_logger(__name__)

class MathPal:
    def __init__(self, model_id: str, device: str = "auto"):
        self.model, self.processor = self._load_model(model_id)
        self.processor = get_chat_template(self.processor, "gemma-3n")
        FastModel.for_inference(self.model)
        
    def _load_model(self, model_id: str):
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            dtype=None,  # Auto-detect
            max_seq_length=settings.MAX_INPUT_TOKENS,
            load_in_4bit=True,
            load_in_8bit=False
        )
        return model, tokenizer

    @opik.track(name="inference_pipeline.format_inference_input")
    def _format_inference_input(self, question: str) -> List[Dict[str, Any]]:
        """
        Format a single question for inference.
        
        Args:
            question: Input question text
            
        Returns:
            Formatted messages for chat template
        """
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": question,
            }]
        }]
            
        return messages


    @opik.track(name="inference_pipeline.generate")
    def generate(self, question: str, sample_for_evaluation: bool = False) -> str:
        messages = self._format_inference_input(question)
        input_ids = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True, return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        
        response = self.model.generate(
            **input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64
        )
        answer = self.processor.batch_decode(response, skip_special_tokens=False)
        
        opik_context.update_current_trace(
            tags=["mathpal_generate"],
            metadata={
                "question": question,
                "response": answer,
                "model_id": settings.MODEL_ID,
            }
        )
        answer = {"answer": answer, "question": question}
        
        return answer
    
if __name__ == "__main__":
    mathpal = MathPal(model_id=settings.MODEL_ID)
    question = "What is the sum of 1 and 2?"
    pprint.pprint(mathpal.generate(question, sample_for_evaluation=True))
        