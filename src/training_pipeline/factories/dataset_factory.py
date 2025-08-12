"""Factory for creating and processing datasets."""

from typing import Dict, Any, Optional
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
    def _load_raw_dataset(config: ConfigManager) -> DatasetDict:
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
                        config: ConfigManager, 
                        tokenizer: Any,
                        is_training: bool = True) -> Dataset:
        """Process dataset with tokenization and formatting for SFTTrainer compatibility."""
        try:
            logger.info(f"📊 Processing dataset with {len(dataset)} samples")
            
            # Apply dataset-specific preprocessing first to create conversation format
            dataset_func_name = config.dataset.name.replace("/", "_").replace("-", "_")
            if hasattr(DatasetFactory, f'_preprocess_{dataset_func_name}'):
                preprocess_func = getattr(DatasetFactory, f'_preprocess_{dataset_func_name}')
                dataset = preprocess_func(dataset, config)
                logger.info(f"✅ Applied preprocessing function: _preprocess_{dataset_func_name}")
                logger.info(f"📊 After preprocessing: {len(dataset)} samples")
            
            # Apply chat template to conversations like working notebook
            if "conversations" in dataset.column_names:
                logger.info("📝 Applying Gemma-3n chat template to conversations...")
                
                def formatting_prompts_func(examples):
                    """Format conversations using chat template like working notebook."""
                    convos = examples["conversations"]
                    texts = []
                    for convo in convos:
                        try:
                            # Apply chat template and remove <bos> prefix like notebook
                            text = tokenizer.apply_chat_template(
                                convo, 
                                tokenize=False, 
                                add_generation_prompt=False
                            ).removeprefix('<bos>')
                            texts.append(text)
                        except Exception as e:
                            logger.warning(f"Failed to apply chat template: {e}")
                            texts.append("")  # Empty text to be filtered out
                    return {"text": texts}
                
                # Apply formatting to create text field
                # CRITICAL: num_proc=1 required for tokenizer.apply_chat_template
                dataset = dataset.map(
                    formatting_prompts_func, 
                    batched=True,
                    num_proc=1,  # MUST be 1 - tokenizer not thread-safe 
                    desc="Applying chat template"
                )
                
                logger.info("✅ Chat template applied successfully")
                logger.info(f"📊 After chat template: {len(dataset)} samples")
            
            # Check if dataset has the required text field after preprocessing
            if config.dataset.text_field not in dataset.column_names:
                if "text" in dataset.column_names:
                    # Rename text column
                    dataset = dataset.rename_column("text", config.dataset.text_field)
                    logger.info(f"📋 Renamed 'text' column to '{config.dataset.text_field}'")
                else:
                    raise DatasetError(f"Dataset missing required field '{config.dataset.text_field}' after preprocessing")
            
            logger.info(f"📋 Dataset format verified. Columns: {dataset.column_names}")
            
            # Keep only text field for SFTTrainer compatibility
            columns_to_keep = [config.dataset.text_field]
            columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

            if columns_to_remove:
                logger.info(f"🧹 Removing extra columns (keeping only {config.dataset.text_field}): {columns_to_remove}")
                dataset = dataset.remove_columns(columns_to_remove)
                logger.info(f"📊 After column removal: {len(dataset)} samples")
            
            # Filter out empty or invalid text samples
            original_size = len(dataset)
            logger.info(f"🔍 Starting filter with {original_size} samples")
            
            dataset = dataset.filter(
                lambda x: x[config.dataset.text_field] is not None and 
                         len(str(x[config.dataset.text_field]).strip()) > 0,
                desc="Filtering empty text samples",
                num_proc=1  # MUST be 1 - prevents silent failures in multiprocessing
            )
            
            filtered_size = len(dataset)

            if filtered_size < original_size:
                logger.info(f"🧹 Filtered out {original_size - filtered_size} empty samples")
            else:
                logger.info(f"✅ All {filtered_size} samples passed filter")
            
            # Tokenize dataset if needed for length validation
            if config.dataset.max_length and config.dataset.max_length > 0:
                logger.info(f"🔍 Applying length filter with max_length={config.dataset.max_length}")
                dataset = DatasetFactory._filter_by_length(dataset, config, tokenizer)
                logger.info(f"📊 After length filter: {len(dataset)} samples")
            
            logger.info(f"✅ Dataset processed: {len(dataset)} samples, columns: {dataset.column_names}")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Dataset processing failed: {e}")
            raise DatasetError(f"Failed to process dataset: {e}")
    
    @staticmethod
    def _filter_by_length(dataset: Dataset, 
                         config: ConfigManager, 
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
                                                                config: ConfigManager) -> Dataset:
        """Preprocess Vietnamese 6th grade exam dataset with Gemma-3n chat template."""
        try:
            logger.info("🇻🇳 Applying Vietnamese math dataset preprocessing with Gemma-3n chat template...")
            
            def format_vietnamese_math_conversations(example):
                """Format example as conversations for Gemma-3n chat template."""
                question = example.get("question", "")
                solution = example.get("solution", "")
                
                if question and solution:
                    # Create conversation format like working notebook
                    conversations = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": question.strip()}]
                        },
                        {
                            "role": "assistant", 
                            "content": [{"type": "text", "text": solution.strip()}]
                        }
                    ]
                    return {"conversations": conversations}
                else:
                    # Return empty to be filtered out
                    return {"conversations": []}
            
            # Convert to conversation format like working notebook
            logger.info("Converting to conversation format...")
            conv_dataset = dataset.map(
                format_vietnamese_math_conversations,
                num_proc=config.dataset.num_proc,
                desc="Converting to conversations"
            )
            
            # Filter out empty conversations
            conv_dataset = conv_dataset.filter(
                lambda x: len(x["conversations"]) > 0,
                num_proc=config.dataset.num_proc,
                desc="Filtering empty conversations"
            )
            
            logger.info(f"Conversation dataset size: {len(conv_dataset)}")
            return conv_dataset
            
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
    def get_dataset_info(config: ConfigManager) -> Dict[str, Any]:
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
