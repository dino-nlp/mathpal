"""Factory for creating and processing datasets."""

from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict

from ..core.exceptions import DatasetError
from ..core.enhanced_config import ComprehensiveTrainingConfig
from ..utils import get_logger

logger = get_logger()


class DatasetFactory:
    """Factory for creating and processing datasets."""
    
    @staticmethod
    def create_dataset(config: ComprehensiveTrainingConfig, tokenizer: Any) -> Dict[str, Any]:
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
        try:
            logger.info(f"📊 Loading dataset: {config.dataset.name}")
            
            # Load dataset from HuggingFace Hub or local path
            datasets = DatasetFactory._load_raw_dataset(config)
            
            # Process datasets
            processed_datasets = {}
            
            # Process training dataset
            if config.dataset.train_split in datasets:
                train_dataset = datasets[config.dataset.train_split]
                processed_datasets["train"] = DatasetFactory._process_dataset(
                    train_dataset, config, tokenizer, is_training=True
                )
                logger.info(f"✅ Training dataset: {len(processed_datasets['train'])} samples")
            else:
                raise DatasetError(f"Training split '{config.dataset.train_split}' not found in dataset")
            
            # Process evaluation dataset if available
            if config.dataset.test_split and config.dataset.test_split in datasets:
                eval_dataset = datasets[config.dataset.test_split]
                processed_datasets["eval"] = DatasetFactory._process_dataset(
                    eval_dataset, config, tokenizer, is_training=False
                )
                logger.info(f"✅ Evaluation dataset: {len(processed_datasets['eval'])} samples")
            
            # Preview dataset
            DatasetFactory._preview_dataset(processed_datasets["train"], num_samples=2)
            
            return processed_datasets
            
        except Exception as e:
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f"Failed to create dataset: {e}")
    
    @staticmethod
    def _load_raw_dataset(config: ComprehensiveTrainingConfig) -> DatasetDict:
        """Load raw dataset from source."""
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
                        config: ComprehensiveTrainingConfig, 
                        tokenizer: Any,
                        is_training: bool = True) -> Dataset:
        """Process dataset with tokenization and formatting."""
        try:
            # Apply dataset-specific preprocessing
            if hasattr(DatasetFactory, f'_preprocess_{config.dataset.name.replace("/", "_").replace("-", "_")}'):
                preprocess_func = getattr(DatasetFactory, f'_preprocess_{config.dataset.name.replace("/", "_").replace("-", "_")}')
                dataset = preprocess_func(dataset, config)
            
            # Check if dataset has the required text field
            if config.dataset.text_field not in dataset.column_names:
                if "text" in dataset.column_names:
                    # Rename text column
                    dataset = dataset.rename_column("text", config.dataset.text_field)
                else:
                    raise DatasetError(f"Dataset missing required field '{config.dataset.text_field}'")
            
            # Filter out empty or invalid samples
            original_size = len(dataset)
            dataset = dataset.filter(
                lambda x: x[config.dataset.text_field] is not None and 
                         len(str(x[config.dataset.text_field]).strip()) > 0
            )
            filtered_size = len(dataset)
            
            if filtered_size < original_size:
                logger.info(f"🧹 Filtered out {original_size - filtered_size} empty samples")
            
            # Tokenize dataset if needed for length validation
            if config.dataset.max_length:
                dataset = DatasetFactory._filter_by_length(dataset, config, tokenizer)
            
            return dataset
            
        except Exception as e:
            raise DatasetError(f"Failed to process dataset: {e}")
    
    @staticmethod
    def _filter_by_length(dataset: Dataset, 
                         config: ComprehensiveTrainingConfig, 
                         tokenizer: Any) -> Dataset:
        """Filter dataset by maximum length."""
        try:
            logger.info(f"🔍 Filtering by max length: {config.dataset.max_length}")
            
            def tokenize_and_check_length(example):
                tokens = tokenizer(
                    example[config.dataset.text_field],
                    truncation=False,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                return {"length": len(tokens["input_ids"])}
            
            # Add length column
            dataset = dataset.map(
                tokenize_and_check_length,
                num_proc=config.dataset.num_proc,
                desc="Computing lengths"
            )
            
            # Filter by length
            original_size = len(dataset)
            dataset = dataset.filter(lambda x: x["length"] <= config.dataset.max_length)
            filtered_size = len(dataset)
            
            # Remove length column
            dataset = dataset.remove_columns(["length"])
            
            logger.info(f"📏 Length filtering: {original_size} → {filtered_size} samples")
            
            return dataset
            
        except Exception as e:
            raise DatasetError(f"Failed to filter dataset by length: {e}")
    
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
    
    # Dataset-specific preprocessing functions
    @staticmethod
    def _preprocess_ngohongthai_exam_sixth_grade_instruct_dataset(dataset: Dataset, 
                                                                config: ComprehensiveTrainingConfig) -> Dataset:
        """Preprocess Vietnamese 6th grade exam dataset."""
        try:
            logger.info("🇻🇳 Applying Vietnamese math dataset preprocessing...")
            
            def format_vietnamese_math(example):
                # Vietnamese math-specific formatting
                text = example.get("text", "")
                
                # Check if we need to format from different field combinations
                if not text or not text.strip():
                    # Try multiple field combinations
                    question = example.get("question", "")
                    solution = example.get("solution", "")
                    instruction = example.get("instruction", "")
                    output = example.get("output", "")
                    
                    # First priority: question/solution (ngohongthai dataset format)
                    if question and solution:
                        formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là một trợ lý giáo dục chuyên về toán học cho học sinh lớp 6 tại Việt Nam. Hãy giải thích chi tiết và dễ hiểu.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{solution}<|eot_id|><|end_of_text|>"""
                        return {"text": formatted_text}
                    
                    # Second priority: instruction/output (standard format)
                    elif instruction and output:
                        formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là một trợ lý giáo dục chuyên về toán học cho học sinh lớp 6 tại Việt Nam. Hãy giải thích chi tiết và dễ hiểu.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|><|end_of_text|>"""
                        return {"text": formatted_text}
                    else:
                        # If no valid field combination, return empty to be filtered out
                        return {"text": ""}
                
                return {"text": text}
            
            dataset = dataset.map(
                format_vietnamese_math,
                num_proc=config.dataset.num_proc,
                desc="Formatting Vietnamese math"
            )
            
            return dataset
            
        except Exception as e:
            logger.warning(f"Failed to apply Vietnamese math preprocessing: {e}")
            return dataset
    
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
    
    @staticmethod
    def get_dataset_info(config: ComprehensiveTrainingConfig) -> Dict[str, Any]:
        """Get information about dataset without loading it."""
        try:
            # Try to get dataset info
            from datasets import get_dataset_infos
            
            infos = get_dataset_infos(config.dataset.name)
            return {
                "name": config.dataset.name,
                "splits": list(infos.keys()) if infos else ["unknown"],
                "features": infos.get("default", {}).get("features", {}) if infos else {},
                "supported": True
            }
            
        except Exception as e:
            return {
                "name": config.dataset.name,
                "error": str(e),
                "supported": False
            }
