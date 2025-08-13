"""Custom exceptions for training pipeline."""

from typing import List, Optional


class TrainingPipelineError(Exception):
    """Base exception for training pipeline."""
    pass


class ValidationError(TrainingPipelineError):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class ModelError(TrainingPipelineError):
    """Raised when model operations fail."""
    pass


class UnsupportedModelError(ModelError):
    """Raised when trying to use an unsupported model."""
    pass


class DatasetError(TrainingPipelineError):
    """Raised when dataset operations fail."""
    pass


class TrainingError(TrainingPipelineError):
    """Raised when training fails."""
    pass


class MemoryError(TrainingPipelineError):
    """Raised when memory requirements exceed available resources."""
    pass


class ConfigurationError(TrainingPipelineError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, errors: Optional[List[ValidationError]] = None):
        self.errors = errors or []
        super().__init__(message)


class ExperimentError(TrainingPipelineError):
    """Raised when experiment tracking fails."""
    pass


class CheckpointError(TrainingPipelineError):
    """Raised when checkpoint operations fail."""
    pass


class InferenceError(TrainingPipelineError):
    """Raised when inference fails."""
    pass
