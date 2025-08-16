"""
Dataset manager for the evaluation pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass

from ..config import ConfigManager
from ..utils import (
    DatasetError,
    ValidationError,
    get_logger,
    load_yaml_config
)


@dataclass
class EvaluationSample:
    """A single evaluation sample."""
    
    id: str
    question: str
    expected_answer: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
        }

class DatasetManager:
    """
    Manages datasets for evaluation.
    
    Handles loading, validation, and preprocessing of evaluation datasets.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize dataset manager.
        
        Args:
            config: Configuration manager
        """
        self.config_manager = config
        self.dataset_config = self.config_manager.get_dataset_config()
        self.logger = get_logger("DatasetManager")
        self.logger.info("Dataset manager initialized")
    
    def load_dataset(self) -> List[EvaluationSample]:
        """
        Load dataset from file or Hugging Face Hub with validation.
        
        Args:
            dataset_path: Path to dataset file or Hugging Face dataset ID
            
        Returns:
            List of evaluation samples
            
        Raises:
            DatasetError: If dataset cannot be loaded or is invalid
        """
        if self.dataset_config.source == "huggingface":
            raw_samples = self._load_huggingface_dataset(self.dataset_config.dataset_id, f"{self.dataset_config.split}[:self.dataset_config.max_samples]")
        else:
            raise NotImplementedError(f"Dataset source {self.dataset_config.source} not implemented")
        
        # Validate and filter samples
        validated_samples = self._validate_samples(raw_samples)
        
        self.logger.info(f"Loaded {len(validated_samples)} valid samples from {self.dataset_config.source}")
        return validated_samples
    
    def _validate_samples(self, samples: List[EvaluationSample]) -> List[EvaluationSample]:
        """
        Validate and filter samples.
        
        Args:
            samples: List of raw samples
            
        Returns:
            List of validated samples
        """
        validated_samples = []
        invalid_count = 0
        
        for i, sample in enumerate(samples):
            try:
                validated_sample = self._validate_single_sample(sample)
                validated_samples.append(validated_sample)
            except ValidationError as e:
                invalid_count += 1
                self.logger.warning(f"Sample {i} (ID: {sample.id}) validation failed: {e}")
            except Exception as e:
                invalid_count += 1
                self.logger.error(f"Unexpected error validating sample {i} (ID: {sample.id}): {e}")
        
        if invalid_count > 0:
            self.logger.warning(f"Skipped {invalid_count} invalid samples out of {len(samples)} total")
        
        return validated_samples
    
    def _validate_single_sample(self, sample: EvaluationSample) -> EvaluationSample:
        """
        Validate a single sample.
        
        Args:
            sample: Sample to validate
            
        Returns:
            Validated sample
            
        Raises:
            ValidationError: If sample is invalid
        """
        # Check required fields
        if not sample.question or not sample.question.strip():
            raise ValidationError("Question is required and cannot be empty")
        
        # Validate question length
        if len(sample.question) > 10000:  # 10KB limit
            raise ValidationError("Question is too long (max 10KB)")
        
        # Validate answer length if present
        if sample.expected_answer and len(sample.expected_answer) > 50000:  # 50KB limit
            raise ValidationError("Expected answer is too long (max 50KB)")
        
        return sample
    
    def _load_huggingface_dataset(self, dataset_id: str, split: str = "test") -> List[EvaluationSample]:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            dataset_id: Hugging Face dataset ID (e.g., "username/dataset_name")
            split: Dataset split to load (default: "test")
            
        Returns:
            List of evaluation samples
            
        Raises:
            DatasetError: If dataset cannot be loaded
        """
        try:
            from datasets import load_dataset
            
            self.logger.info(f"Loading dataset from Hugging Face: {dataset_id} (split: {split})") 
            # Load dataset from Hugging Face
            dataset = load_dataset(dataset_id, split=split)
            samples = []
            for i, item in enumerate(dataset):
                question = item[self.dataset_config.instruction_column]
                expected_answer = item[self.dataset_config.answer_column]
                
                sample = EvaluationSample(
                    id=item.get("id", str(i)),
                    question=question,
                    expected_answer=expected_answer,
                )
                samples.append(sample)
            
            self.logger.info(f"Loaded {len(samples)} samples from Hugging Face dataset: {dataset_id}")
            return samples
            
        except ImportError:
            raise DatasetError("datasets library not installed. Install with: pip install datasets")
        except Exception as e:
            raise DatasetError(f"Error loading Hugging Face dataset {dataset_id}: {e}")
    
    def validate_dataset(self, samples: List[EvaluationSample]) -> bool:
        """
        Validate dataset samples.
        
        Args:
            samples: List of evaluation samples
            
        Returns:
            True if valid, False otherwise
        """
        if not samples:
            self.logger.error("Dataset is empty")
            return False
        
        for i, sample in enumerate(samples):
            if not sample.question:
                self.logger.error(f"Sample {i} has empty question")
                return False
            
            if not sample.id:
                self.logger.error(f"Sample {i} has empty ID")
                return False
            
            if not sample.expected_answer:
                self.logger.error(f"Sample {i} has empty expected answer")
                return False
            
        self.logger.info(f"Validated {len(samples)} samples")
        return True
    
    def save_dataset(
        self, 
        samples: List[EvaluationSample], 
        output_path: Union[str, Path]
    ) -> None:
        """
        Save dataset to file.
        
        Args:
            samples: List of evaluation samples
            output_path: Path to save dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [sample.to_dict() for sample in samples]
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved {len(samples)} samples to {output_path}")
        except Exception as e:
            raise DatasetError(f"Error saving dataset: {e}")
    