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

from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logger_utils.get_logger(__name__)

class MathPal:
    def __init__(self, model_id: str, device: str = "auto"):
        self.processor, self.model = self._load_model(model_id, device)
        
    def _load_model(self, model_id: str, device: str = "auto"):
        # Select device without relying on Accelerate's infer_auto_device to avoid warnings
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        processor = AutoProcessor.from_pretrained(model_id)

        # Try 4-bit quantization on CUDA to reduce memory usage
        if resolved_device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                )
                model.eval()
                return processor, model
            except Exception:
                # Fallback to FP16 if bitsandbytes is unavailable or model doesn't support 4-bit
                pass

        load_dtype = torch.float16 if resolved_device == "cuda" else "auto"
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
        )
        try:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
        except Exception:
            pass
        model.to(resolved_device)
        model.eval()
        return processor, model

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
        