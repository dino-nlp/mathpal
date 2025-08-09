"""Dataset processing utilities."""

from typing import Optional, Union, Dict, Any
from datasets import load_dataset, Dataset
from .chat_formatter import ChatFormatter


class DatasetProcessor:
    """Handles dataset loading and processing for training."""
    
    def __init__(self, tokenizer):
        """Initialize DatasetProcessor with tokenizer."""
        self.tokenizer = tokenizer
        self.chat_formatter = ChatFormatter(tokenizer)
    
    def load_dataset(
        self, 
        dataset_name: str, 
        split: str = "train",
        subset: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> Dataset:
        """
        Load dataset from HuggingFace Hub or local path.
        
        Args:
            dataset_name: Name or path of the dataset
            split: Dataset split to load
            subset: Optional subset name
            cache_dir: Optional cache directory
            
        Returns:
            Loaded dataset
        """
        print(f"Loading dataset: {dataset_name}, split: {split}")
        
        try:
            if subset:
                dataset = load_dataset(
                    dataset_name, 
                    subset,
                    split=split,
                    cache_dir=cache_dir
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split=split, 
                    cache_dir=cache_dir
                )
            
            print(f"Successfully loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def prepare_dataset(
        self, 
        dataset_name: str, 
        split: str = "train",
        subset: Optional[str] = None,
        apply_chat_template: bool = True,
        cache_dir: Optional[str] = None
    ) -> Dataset:
        """
        Load and prepare dataset for training.
        
        Args:
            dataset_name: Name or path of the dataset
            split: Dataset split to load
            subset: Optional subset name
            apply_chat_template: Whether to apply chat template formatting
            cache_dir: Optional cache directory
            
        Returns:
            Processed dataset ready for training
        """
        # Load raw dataset
        dataset = self.load_dataset(
            dataset_name=dataset_name,
            split=split,
            subset=subset,
            cache_dir=cache_dir
        )
        
        # Apply chat template formatting if requested
        if apply_chat_template:
            dataset = self.chat_formatter.apply_chat_template_to_dataset(dataset)
        
        return dataset
    
    def prepare_datasets(
        self,
        dataset_name: str,
        train_split: str = "train", 
        eval_split: Optional[str] = None,
        subset: Optional[str] = None,
        apply_chat_template: bool = True,
        cache_dir: Optional[str] = None
    ) -> Dict[str, Dataset]:
        """
        Prepare train and optional eval datasets.
        
        Args:
            dataset_name: Name or path of the dataset
            train_split: Training split name
            eval_split: Optional evaluation split name
            subset: Optional subset name
            apply_chat_template: Whether to apply chat template formatting
            cache_dir: Optional cache directory
            
        Returns:
            Dictionary with 'train' and optionally 'eval' datasets
        """
        datasets = {}
        
        # Prepare training dataset
        datasets["train"] = self.prepare_dataset(
            dataset_name=dataset_name,
            split=train_split,
            subset=subset,
            apply_chat_template=apply_chat_template,
            cache_dir=cache_dir
        )
        
        # Prepare evaluation dataset if specified
        if eval_split:
            datasets["eval"] = self.prepare_dataset(
                dataset_name=dataset_name,
                split=eval_split,
                subset=subset,
                apply_chat_template=apply_chat_template,
                cache_dir=cache_dir
            )
        
        return datasets
    
    def preview_dataset(self, dataset: Dataset, num_samples: int = 3) -> None:
        """
        Preview dataset samples.
        
        Args:
            dataset: Dataset to preview
            num_samples: Number of samples to show
        """
        print(f"\n📊 Dataset preview ({len(dataset)} total samples):")
        print(f"Features: {list(dataset.features.keys())}")
        
        for i in range(min(num_samples, len(dataset))):
            print(f"\n--- Sample {i+1} ---")
            sample = dataset[i]
            
            for key, value in sample.items():
                if isinstance(value, str):
                    # Truncate long text
                    preview_text = value[:200] + "..." if len(value) > 200 else value
                    print(f"{key}: {preview_text}")
                else:
                    print(f"{key}: {value}")
    
    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "num_samples": len(dataset),
            "features": list(dataset.features.keys()),
            "feature_types": {k: str(v) for k, v in dataset.features.items()}
        }
        
        # Calculate text length statistics if 'text' field exists
        if 'text' in dataset.features:
            text_lengths = [len(sample['text']) for sample in dataset]
            stats["text_length_stats"] = {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": sum(text_lengths) / len(text_lengths),
                "median": sorted(text_lengths)[len(text_lengths) // 2]
            }
        
        return stats