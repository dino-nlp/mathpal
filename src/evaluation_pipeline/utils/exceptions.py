"""
Custom exceptions for the evaluation pipeline.
"""


class EvaluationPipelineError(Exception):
    """Base exception for evaluation pipeline errors."""
    pass


class ConfigurationError(EvaluationPipelineError):
    """Raised when there's an error in configuration."""
    pass


class ModelError(EvaluationPipelineError):
    """Raised when there's an error with model loading or inference."""
    pass


class DatasetError(EvaluationPipelineError):
    """Raised when there's an error with dataset operations."""
    pass


class MetricsError(EvaluationPipelineError):
    """Raised when there's an error with metrics calculation."""
    pass


class ProviderError(EvaluationPipelineError):
    """Raised when there's an error with LLM providers (OpenRouter, etc.)."""
    pass


class OpikError(EvaluationPipelineError):
    """Raised when there's an error with Opik integration."""
    pass


class ValidationError(EvaluationPipelineError):
    """Raised when input validation fails."""
    pass


class HardwareError(EvaluationPipelineError):
    """Raised when there's an error with hardware requirements."""
    pass


class InferenceError(ModelError):
    """Raised when there's an error during model inference."""
    pass


class BatchProcessingError(EvaluationPipelineError):
    """Raised when there's an error during batch processing."""
    pass


class StreamingError(EvaluationPipelineError):
    """Raised when there's an error during streaming operations."""
    pass
