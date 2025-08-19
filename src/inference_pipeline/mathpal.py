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
        prompt, num_tokens = truncate_text_to_max_tokens(question, settings.MAX_INPUT_TOKENS)
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt,
            }]
        }]
            
        return messages, num_tokens
        
    def _decode_batch_outputs(
        self, 
        processor,
        batch_outputs: torch.Tensor, 
        batch_inputs: Dict[str, Any], 
        return_full_text: bool
    ) -> List[str]:
        """
        Decode batch outputs to text responses.
        
        Args:
            batch_outputs: Model outputs tensor
            batch_inputs: Input dictionary used for generation
            return_full_text: Whether to return full text
            
        Returns:
            List of decoded text responses
        """
        responses = []
        
        if return_full_text:
            # Return full generated text for each item
            for i in range(batch_outputs.shape[0]):
                generated_text = processor.decode(
                    batch_outputs[i], 
                    skip_special_tokens=True
                )
                responses.append(generated_text.strip())
        else:
            # Return only new tokens for each item
            input_length = batch_inputs['input_ids'].shape[1]
            for i in range(batch_outputs.shape[0]):
                new_tokens = batch_outputs[i][input_length:]
                generated_text = processor.decode(
                    new_tokens, 
                    skip_special_tokens=True
                )
                responses.append(generated_text.strip())
        
        return responses
    
    @opik.track(name="inference_pipeline.generate")
    def generate(self, question: str, sample_for_evaluation: bool = False) -> str:
        messages, num_tokens = self._format_inference_input(question)
        input_ids = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True, return_dict=True,
                return_tensors="pt",
        )
        input_ids = input_ids.to(self.model.device, dtype=self.model.dtype)

        # Generate output from the model
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model.generate(**input_ids, max_new_tokens=128)
            else:
                outputs = self.model.generate(**input_ids, max_new_tokens=128)

        # decode and print the output as text
        response = self._decode_batch_outputs(self.processor, outputs, input_ids, return_full_text=False)
        answer = response[0]
        num_answer_tokens = compute_num_tokens(answer)
        
        opik_context.update_current_trace(
            tags=["mathpal_generate"],
            metadata={
                "question": question,
                "response": answer,
                "model_id": settings.MODEL_ID,
                "input_tokens": num_tokens,
                "output_tokens": num_answer_tokens,
                "total_tokens": num_tokens + num_answer_tokens,
            }
        )
        answer = {"answer": answer, "question": question}
        
        return answer
    
if __name__ == "__main__":
    mathpal = MathPal(model_id=settings.MODEL_ID)
    question = "What is the sum of 1 and 2?"
    pprint.pprint(mathpal.generate(question, sample_for_evaluation=True))
        