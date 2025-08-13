"""Chat template formatting utilities."""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datasets import Dataset

if TYPE_CHECKING:
    from training_pipeline.config import DatasetConfigSection


class ChatFormatter:
    """Handles chat template formatting for conversational datasets."""
    
    def __init__(self, tokenizer: Any, data_config: "DatasetConfigSection"):
        """Initialize ChatFormatter with tokenizer."""
        self.tokenizer = tokenizer
        self.data_config = data_config
    
    def process_sample(self, sample: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert a sample with question/solution to conversation format.
        
        Args:
            sample: Dict with 'question' and 'solution' keys
            
        Returns:
            Dict with 'conversations' key containing formatted conversation
        """
        conversations = [
            {
                "role": "user",
                "content": [{"type": "text", "text": sample[self.data_config.instruction_column]}]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": sample[self.data_config.answer_column]}]
            }
        ]
        
        return {"conversations": conversations}
    
    def formatting_prompts_func(self, examples: Dict[str, List]) -> Dict[str, List[str]]:
        """
        Apply chat template to batch of examples.
        
        Args:
            examples: Batch of examples with 'conversations' key
            
        Returns:
            Dict with 'text' key containing formatted texts
        """
        convos = examples["conversations"]
        texts = [
            self.tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            ).removeprefix('<bos>') 
            for convo in convos
        ]
        return {self.data_config.text_field: texts}
    
    def apply_chat_template_to_dataset(self, dataset: Dataset) -> Dataset:
        """
        Apply chat template formatting to entire dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Formatted dataset with 'text' field
        """
        # First convert samples to conversation format
        processed_dataset = dataset.map(self.process_sample)
        
        # Then apply chat template
        formatted_dataset = processed_dataset.map(
            self.formatting_prompts_func, 
            batched=True
        )
        
        return formatted_dataset
    
    def format_inference_input(self, question: str) -> List[Dict[str, Any]]:
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
    
    def prepare_inference_inputs(self, question: str, device: str = "cuda") -> Dict[str, Any]:
        """
        Prepare inputs for model inference.
        
        Args:
            question: Input question
            device: Target device
            
        Returns:
            Tokenized inputs ready for model
        """
        messages = self.format_inference_input(question)
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to(device)
        
        return inputs