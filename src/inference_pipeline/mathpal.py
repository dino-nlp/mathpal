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

# Disable tokenizers parallelism to avoid deadlocks during forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable TorchDynamo to avoid FX symbolic tracing conflicts with Unsloth
# This prevents the "Detected that you are using FX to symbolically trace a dynamo-optimized function" error
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True  # Completely disable TorchDynamo

# Set environment variables to prevent compilation issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = "off"

logger = logger_utils.get_logger(__name__)

class MathPal:
    def __init__(self, model_id: str, device: str = "auto"):
        self.model, self.processor = self._load_model(model_id)
        self.processor = get_chat_template(self.processor, "gemma-3n")
        
        # Use safe inference mode
        try:
            FastModel.for_inference(self.model)
            logger.info("✅ Model prepared for inference (safe mode)")
        except Exception as e:
            logger.warning(f"⚠️  Could not prepare model for inference: {e}")
            logger.info("🔄 Continuing with standard model...")
        
        # Set model to evaluation mode for more stable inference
        self.model.eval()
        
    def _load_model(self, model_id: str):
        try:
            # Try to load with GPU first - using safe configuration
            logger.info("🔄 Attempting to load model with GPU (safe mode)...")
            model, processor = FastModel.from_pretrained(
                model_name=model_id,
                dtype=None,  # Auto-detect
                max_seq_length=settings.MAX_INPUT_TOKENS,
                load_in_4bit=True,
                load_in_8bit=False,
                # Use safe device mapping
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            logger.info("✅ Model loaded successfully with GPU (safe mode)")
            return model, processor
            
        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            if "Some modules are dispatched on the CPU" in error_msg or "FX to symbolically trace" in error_msg:
                logger.warning("⚠️  GPU RAM insufficient or FX tracing conflict, attempting CPU offload...")
                try:
                    # Try with CPU offload enabled
                    model, processor = FastModel.from_pretrained(
                        model_name=model_id,
                        dtype=None,  # Auto-detect
                        max_seq_length=settings.MAX_INPUT_TOKENS,
                        load_in_4bit=True,
                        load_in_8bit=False,
                        device_map="auto",  # Enable automatic device mapping
                        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload
                    )
                    logger.info("✅ Model loaded successfully with CPU offload")
                    return model, processor
                    
                except Exception as cpu_error:
                    logger.error(f"❌ CPU offload failed: {cpu_error}")
                    logger.info("🔄 Attempting to load with 8-bit quantization...")
                    
                    # Try with 8-bit quantization instead of 4-bit
                    try:
                        model, processor = FastModel.from_pretrained(
                            model_name=model_id,
                            dtype=None,  # Auto-detect
                            max_seq_length=settings.MAX_INPUT_TOKENS,
                            load_in_4bit=False,
                            load_in_8bit=True,
                            device_map="auto"
                        )
                        logger.info("✅ Model loaded successfully with 8-bit quantization")
                        return model, processor
                        
                    except Exception as bit8_error:
                        logger.error(f"❌ 8-bit quantization failed: {bit8_error}")
                        logger.info("🔄 Attempting to load without quantization...")
                        
                        # Last resort: load without quantization
                        try:
                            model, processor = FastModel.from_pretrained(
                                model_name=model_id,
                                dtype=None,  # Auto-detect
                                max_seq_length=settings.MAX_INPUT_TOKENS,
                                load_in_4bit=False,
                                load_in_8bit=False,
                                device_map="auto"
                            )
                            logger.info("✅ Model loaded successfully without quantization")
                            return model, processor
                            
                        except Exception as final_error:
                            logger.error(f"❌ All loading methods failed: {final_error}")
                            raise RuntimeError(f"Failed to load model {model_id}. Please check your GPU memory and try again.")
            else:
                # Re-raise if it's not a memory or FX tracing issue
                raise e

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
    def generate(self, question: str) -> str:
        messages = self._format_inference_input(question)
        input_ids = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True, return_dict=True,
            return_tensors="pt",
        )
        
        # Determine device for input_ids based on model device
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            # For models with device mapping, use the first device
            device_map = self.model.hf_device_map
            if device_map:
                device = list(device_map.values())[0]
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Move input_ids to the appropriate device
        input_ids = {k: v.to(device) if hasattr(v, 'to') else v for k, v in input_ids.items()}
        
        # Use torch.no_grad() for inference to save memory and prevent gradient computation
        with torch.no_grad():
            response = self.model.generate(
                **input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                pad_token_id=self.processor.eos_token_id,
                eos_token_id=self.processor.eos_token_id,
            )
        answer = self.processor.batch_decode(response, skip_special_tokens=False)[0]
        
        # Clear CUDA cache to prevent memory accumulation during evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        opik_context.update_current_trace(
            tags=["mathpal_generate"],
            metadata={
                "question": question,
                "response": answer,
                "model_id": settings.MODEL_ID,
                "device": str(device),
            }
        )
        answer = {"answer": answer, "question": question}
        
        return answer
    
if __name__ == "__main__":
    mathpal = MathPal(model_id=settings.MODEL_ID)
    question = "What is the sum of 1 and 2?"
    pprint.pprint(mathpal.generate(question))
        