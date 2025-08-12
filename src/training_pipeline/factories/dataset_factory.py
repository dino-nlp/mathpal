"""Factory for creating and processing datasets."""

from typing import Dict, Any, Optional, List
from datasets import load_dataset, Dataset, DatasetDict

from training_pipeline.utils.exceptions import DatasetError
from training_pipeline.config.config_manager import ConfigManager
from training_pipeline.utils import get_logger

logger = get_logger()


class DatasetFactory:
    """Factory for creating and processing datasets."""
    
    @staticmethod
    def create_dataset(config: ConfigManager, tokenizer: Any) -> Dict[str, Any]:
        """
        Create and process dataset based on configuration.
        
        Args:
            config: Training configuration
            tokenizer: Tokenizer for preprocessing
            
        Returns:
            Dictionary containing train and optional eval datasets
            
        Raises:
            DatasetError: If dataset creation fails
        """
        logger.info(f"📊 Loading dataset: {config.dataset.name}")
        # Load dataset from HuggingFace Hub or local path
        datasets = DatasetFactory._load_raw_dataset(config)
        train_dataset = datasets[config.dataset.train_split]
        # Process datasets
        processed_datasets = DatasetFactory._process_dataset(
                train_dataset, config, tokenizer
            )
        processed_datasets = DatasetFactory.create_eval_dataset(processed_datasets)
        logger.info(f"✅ Training dataset: {len(processed_datasets['train'])} samples")
        logger.info(f"✅ Evaluation dataset: {len(processed_datasets['eval'])} samples")
            # Preview dataset
        DatasetFactory._preview_dataset(processed_datasets["train"], num_samples=2)
        return processed_datasets
            
    
    @staticmethod
    def _load_raw_dataset(config: ConfigManager) -> DatasetDict:
        """Load raw dataset from source."""
        # TODO: Load dataset from comet artifact
        try:
            # Check if it's a local path or HuggingFace dataset
            dataset_name = config.dataset.name
            
            if dataset_name.startswith('/') or dataset_name.startswith('./'):
                # Local dataset
                logger.info(f"Loading local dataset from: {dataset_name}")
                dataset = load_dataset("json", data_files=dataset_name)
            else:
                # HuggingFace Hub dataset
                logger.info(f"Loading HuggingFace dataset: {dataset_name}")
                dataset = load_dataset(dataset_name)
            
            return dataset
            
        except Exception as e:
            raise DatasetError(f"Failed to load dataset {config.dataset.name}: {e}")
    
    @staticmethod
    def _process_dataset(dataset: Dataset, 
                        config: ConfigManager, 
                        tokenizer: Any) -> Dataset:
        """Process dataset with tokenization and formatting for SFTTrainer compatibility."""
        def process_sample(sample: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
            # Create conversation
            conversations = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": sample["question"]}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["solution"]}]
                }
            ]

            return {"conversations": conversations}
        
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
            return { "text" : texts, }
        
        processed_dataset = dataset.map(process_sample)
        formated_dataset = processed_dataset.map(formatting_prompts_func, batched=True)
        return formated_dataset
        
    
    @staticmethod
    def _preview_dataset(dataset: Dataset, num_samples: int = 2):
        """Preview dataset samples."""
        try:
            logger.info("📖 Dataset preview:")
            
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                
                # Extract text content (handle different formats)
                text_content = sample.get("text", "")
                if isinstance(text_content, dict):
                    text_content = str(text_content)
                
                # Truncate for preview
                preview_text = text_content[:200] + "..." if len(text_content) > 200 else text_content
                
                logger.info(f"   Sample {i+1}: {preview_text}")
                logger.info("   " + "-" * 50)
                
        except Exception as e:
            logger.warning(f"Failed to preview dataset: {e}")
    
    @staticmethod
    def create_eval_dataset(train_dataset: Dataset, 
                          test_ratio: float = 0.1,
                          seed: int = 42) -> Dict[str, Dataset]:
        """Split training dataset to create evaluation set."""
        try:
            logger.info(f"✂️ Creating eval split with ratio: {test_ratio}")
            
            split_dataset = train_dataset.train_test_split(
                test_size=test_ratio,
                seed=seed,
                shuffle=True
            )
            
            return {
                "train": split_dataset["train"],
                "eval": split_dataset["test"]
            }
            
        except Exception as e:
            raise DatasetError(f"Failed to create eval split: {e}")
    
