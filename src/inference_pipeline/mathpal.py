import pprint
import opik

from typing import Dict, List, Any, Optional
# from config import settings
# from core import logger_utils
# from core.opik_utils import add_to_dataset_with_sampling
# from opik import opik_context
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# logger = logger_utils.get_logger(__name__)

class MathPal:
    def __init__(self, model_id: str, device: str = "auto"):
        self.processor, self.model = self._load_model(model_id, device)
        
    def _load_model(self, model_id: str, device: str = "auto"):
        processor = AutoProcessor.from_pretrained(model_id, device_map=device)
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="auto" ,device_map=device)
        model.eval()
        return processor, model

    def _format_inference_input(self, question: str) -> List[Dict[str, Any]]:
        """
        Format a single question for inference.
        
        Args:
            question: Input question text
            
        Returns:
            Formatted messages for chat template
        """
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": question,
            }]
        }]
        
    def _decode_batch_outputs(
        self, 
        processor: AutoProcessor,
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
    
    def generate(self, question: str) -> str:
        messages = self._format_inference_input(question)
        input_ids = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True, return_dict=True,
                return_tensors="pt",
        )
        input_ids = input_ids.to(self.model.device, dtype=self.model.dtype)

        # Generate output from the model
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, max_new_tokens=128)

        # decode and print the output as text
        response = self._decode_batch_outputs(self.processor, outputs, input_ids, return_full_text=False)
        return response[0]
    
if __name__ == "__main__":
    mathpal = MathPal(model_id="unsloth/gemma-3n-E2B-it")
    # mathpal = MathPal(model_id="ngohongthai/gemma-3n-E2B-it-mathpal-grade6-vi-fp16")
    question = "What is the sum of 1 and 2?"
    pprint.pprint(mathpal.generate(question))
        