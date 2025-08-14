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
    context: Optional[str] = None
    expected_answer: Optional[str] = None
    grade_level: Optional[str] = None
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "expected_answer": self.expected_answer,
            "grade_level": self.grade_level,
            "subject": self.subject,
            "difficulty": self.difficulty,
            "metadata": self.metadata or {}
        }


@dataclass
class DatasetMetadata:
    """Information about a dataset."""
    
    name: str
    description: str
    num_samples: int
    grade_levels: List[str]
    subjects: List[str]
    difficulties: List[str]
    source: str
    version: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "num_samples": self.num_samples,
            "grade_levels": self.grade_levels,
            "subjects": self.subjects,
            "difficulties": self.difficulties,
            "source": self.source,
            "version": self.version,
            "metadata": self.metadata
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
        self.config = config
        self.logger = get_logger("DatasetManager")
        
        # Default dataset paths
        self.default_datasets = {
            "vietnamese_math_grade5": "data/evaluation/vietnamese_math_grade5.json",
            "vietnamese_math_grade6": "data/evaluation/vietnamese_math_grade6.json",
            "mixed_grade5_6": "data/evaluation/mixed_grade5_6.json"
        }
        
        self.logger.info("Dataset manager initialized")
    
    def load_dataset(self, dataset_path: Union[str, Path]) -> List[EvaluationSample]:
        """
        Load dataset from file or Hugging Face Hub with validation.
        
        Args:
            dataset_path: Path to dataset file or Hugging Face dataset ID
            
        Returns:
            List of evaluation samples
            
        Raises:
            DatasetError: If dataset cannot be loaded or is invalid
        """
        dataset_path = str(dataset_path)
        
        # Check if it's a Hugging Face dataset ID
        if "/" in dataset_path and not Path(dataset_path).exists():
            raw_samples = self._load_huggingface_dataset(dataset_path)
        else:
            # Load from local file
            dataset_path = Path(dataset_path)
            
            if not dataset_path.exists():
                raise DatasetError(f"Dataset file does not exist: {dataset_path}")
            
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                raw_samples = []
                for item in data:
                    sample = EvaluationSample(
                        id=item.get("id", str(len(raw_samples))),
                        question=item["question"],
                        context=item.get("context"),
                        expected_answer=item.get("expected_answer"),
                        grade_level=item.get("grade_level"),
                        subject=item.get("subject"),
                        difficulty=item.get("difficulty"),
                        metadata=item.get("metadata", {})
                    )
                    raw_samples.append(sample)
                
            except json.JSONDecodeError as e:
                raise DatasetError(f"Error parsing JSON dataset: {e}")
            except Exception as e:
                raise DatasetError(f"Error loading dataset: {e}")
        
        # Validate and filter samples
        validated_samples = self._validate_samples(raw_samples)
        
        self.logger.info(f"Loaded {len(validated_samples)} valid samples from {dataset_path}")
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
        
        # Validate grade level if present
        if sample.grade_level:
            valid_grades = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
            if sample.grade_level not in valid_grades:
                raise ValidationError(f"Invalid grade level: {sample.grade_level}")
        
        # Validate subject if present
        if sample.subject:
            valid_subjects = ["Toán", "Văn", "Tiếng Việt", "Khoa học", "Lịch sử", "Địa lý"]
            if sample.subject not in valid_subjects:
                raise ValidationError(f"Invalid subject: {sample.subject}")
        
        # Validate difficulty if present
        if sample.difficulty:
            valid_difficulties = ["Dễ", "Trung bình", "Khó"]
            if sample.difficulty not in valid_difficulties:
                raise ValidationError(f"Invalid difficulty: {sample.difficulty}")
        
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
            
            # Get field mapping from config
            field_mapping = self.config.config.dataset.field_mapping
            
            # Load dataset from Hugging Face
            dataset = load_dataset(dataset_id, split=split)
            
            samples = []
            for i, item in enumerate(dataset):
                # Map fields using config
                question = self._get_field_value(item, [field_mapping["question"]] if isinstance(field_mapping["question"], str) else field_mapping["question"])
                context = self._get_field_value(item, [field_mapping["context"]] if isinstance(field_mapping["context"], str) else field_mapping["context"])
                expected_answer = self._get_field_value(item, [field_mapping["answer"]] if isinstance(field_mapping["answer"], str) else field_mapping["answer"])
                
                # Handle optional fields with defaults
                grade_level = self._get_field_value(item, [field_mapping["grade_level"]] if isinstance(field_mapping["grade_level"], str) else field_mapping["grade_level"], default="5")
                subject = self._get_field_value(item, [field_mapping["subject"]] if isinstance(field_mapping["subject"], str) else field_mapping["subject"], default="Toán")
                difficulty = self._get_field_value(item, [field_mapping["difficulty"]] if isinstance(field_mapping["difficulty"], str) else field_mapping["difficulty"], default="Trung bình")
                
                sample = EvaluationSample(
                    id=item.get("id", str(i)),
                    question=question,
                    context=context,
                    expected_answer=expected_answer,
                    grade_level=grade_level,
                    subject=subject,
                    difficulty=difficulty,
                    metadata={
                        "source": "huggingface",
                        "dataset_id": dataset_id,
                        "split": split,
                        **item.get("metadata", {})
                    }
                )
                samples.append(sample)
            
            self.logger.info(f"Loaded {len(samples)} samples from Hugging Face dataset: {dataset_id}")
            return samples
            
        except ImportError:
            raise DatasetError("datasets library not installed. Install with: pip install datasets")
        except Exception as e:
            raise DatasetError(f"Error loading Hugging Face dataset {dataset_id}: {e}")
    
    def _get_field_value(self, item: Dict[str, Any], field_names: List[str], default: str = "") -> str:
        """
        Get field value from item using prioritized field names.
        
        Args:
            item: Dataset item
            field_names: List of field names to try (in order of priority)
            default: Default value if no field found
            
        Returns:
            Field value or default
        """
        for field_name in field_names:
            if field_name in item and item[field_name]:
                return str(item[field_name])
        return default
    
    def get_default_dataset(self) -> List[EvaluationSample]:
        """
        Get default dataset for evaluation.
        
        Returns:
            List of evaluation samples
        """
        # Get default dataset from config
        default_source = self.config.config.dataset.source
        default_dataset_id = self.config.config.dataset.dataset_id
        default_split = self.config.config.dataset.split
        
        if default_source == "huggingface":
            try:
                self.logger.info(f"Loading default Hugging Face dataset: {default_dataset_id}")
                return self.load_dataset(default_dataset_id)
            except Exception as e:
                self.logger.warning(f"Failed to load default Hugging Face dataset: {e}")
        
        # Fallback to local files
        for dataset_name, dataset_path in self.default_datasets.items():
            if Path(dataset_path).exists():
                self.logger.info(f"Loading fallback local dataset: {dataset_name}")
                return self.load_dataset(dataset_path)
        
        # If no default dataset exists, create a minimal test dataset
        self.logger.warning("No default dataset found, creating minimal test dataset")
        return self._create_test_dataset()
    
    def load_predefined_dataset(self, dataset_name: str) -> List[EvaluationSample]:
        """
        Load a predefined dataset by name.
        
        Args:
            dataset_name: Name of predefined dataset (e.g., "ngohongthai")
            
        Returns:
            List of evaluation samples
            
        Raises:
            DatasetError: If predefined dataset not found
        """
        predefined_datasets = self.config.config.dataset.predefined
        
        if dataset_name not in predefined_datasets:
            available = list(predefined_datasets.keys())
            raise DatasetError(f"Predefined dataset '{dataset_name}' not found. Available: {available}")
        
        dataset_info = predefined_datasets[dataset_name]
        dataset_id = dataset_info["id"]
        split = dataset_info.get("split", "test")
        
        self.logger.info(f"Loading predefined dataset: {dataset_name} ({dataset_id})")
        return self.load_dataset(dataset_id)
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of predefined dataset (None for default)
            
        Returns:
            Dataset information
        """
        if dataset_name:
            predefined_datasets = self.config.config.dataset.predefined
            if dataset_name in predefined_datasets:
                return predefined_datasets[dataset_name]
            else:
                return {"error": f"Dataset '{dataset_name}' not found"}
        
        # Return default dataset info
        return {
            "source": self.config.config.dataset.source,
            "id": self.config.config.dataset.dataset_id,
            "split": self.config.config.dataset.split,
            "info": getattr(self.config.config.dataset, 'info', {})
        }
    
    def _create_test_dataset(self) -> List[EvaluationSample]:
        """
        Create a minimal test dataset for evaluation.
        
        Returns:
            List of test evaluation samples
        """
        test_samples = [
            EvaluationSample(
                id="test_1",
                question="Tính: 15 + 27 = ?",
                expected_answer="42",
                grade_level="5",
                subject="Toán",
                difficulty="Dễ",
                metadata={"type": "addition"}
            ),
            EvaluationSample(
                id="test_2", 
                question="Một hình chữ nhật có chiều dài 8cm và chiều rộng 6cm. Tính diện tích hình chữ nhật đó.",
                expected_answer="48 cm²",
                grade_level="5",
                subject="Toán",
                difficulty="Trung bình",
                metadata={"type": "area_calculation"}
            ),
            EvaluationSample(
                id="test_3",
                question="Tìm x biết: 3x + 5 = 20",
                expected_answer="x = 5",
                grade_level="6", 
                subject="Toán",
                difficulty="Khó",
                metadata={"type": "equation"}
            )
        ]
        
        self.logger.info(f"Created test dataset with {len(test_samples)} samples")
        return test_samples
    
    def get_current_dataset_info(self) -> DatasetMetadata:
        """
        Get information about the current dataset.
        
        Returns:
            Dataset information
        """
        # For now, return basic info
        # In a real implementation, this would analyze the actual dataset
        return DatasetMetadata(
            name="Vietnamese Math Evaluation Dataset",
            description="Dataset for evaluating Vietnamese math education models",
            num_samples=100,  # Placeholder
            grade_levels=["5", "6"],
            subjects=["Toán"],
            difficulties=["Dễ", "Trung bình", "Khó"],
            source="MathPal Generated",
            version="1.0.0",
            metadata={
                "language": "Vietnamese",
                "target_audience": "Grade 5-6 students",
                "subject_focus": "Mathematics"
            }
        )
    
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
        
        self.logger.info(f"Validated {len(samples)} samples")
        return True
    
    def filter_dataset(
        self, 
        samples: List[EvaluationSample],
        grade_level: Optional[str] = None,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> List[EvaluationSample]:
        """
        Filter dataset based on criteria.
        
        Args:
            samples: List of evaluation samples
            grade_level: Filter by grade level
            subject: Filter by subject
            difficulty: Filter by difficulty
            max_samples: Maximum number of samples to return
            
        Returns:
            Filtered list of samples
        """
        filtered_samples = samples
        
        if grade_level:
            filtered_samples = [s for s in filtered_samples if s.grade_level == grade_level]
        
        if subject:
            filtered_samples = [s for s in filtered_samples if s.subject == subject]
        
        if difficulty:
            filtered_samples = [s for s in filtered_samples if s.difficulty == difficulty]
        
        if max_samples and len(filtered_samples) > max_samples:
            filtered_samples = filtered_samples[:max_samples]
        
        self.logger.info(f"Filtered dataset: {len(samples)} -> {len(filtered_samples)} samples")
        return filtered_samples
    
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
    
    def create_dataset_from_questions(
        self, 
        questions: List[str],
        grade_level: str = "5",
        subject: str = "Toán"
    ) -> List[EvaluationSample]:
        """
        Create dataset from list of questions.
        
        Args:
            questions: List of question strings
            grade_level: Grade level for all questions
            subject: Subject for all questions
            
        Returns:
            List of evaluation samples
        """
        samples = []
        
        for i, question in enumerate(questions):
            sample = EvaluationSample(
                id=f"generated_{i+1}",
                question=question,
                grade_level=grade_level,
                subject=subject,
                difficulty="Trung bình",  # Default difficulty
                metadata={"source": "generated"}
            )
            samples.append(sample)
        
        self.logger.info(f"Created dataset with {len(samples)} questions")
        return samples
