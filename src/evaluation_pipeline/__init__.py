"""
MathPal Evaluation Pipeline
==========================

A comprehensive evaluation pipeline for Vietnamese math education AI models.
Features Gemma 3N inference with MatFormer optimization, Opik evaluation,
and OpenRouter integration.

Key Components:
- Gemma 3N Inference Engine with MatFormer optimization
- Opik-based evaluation with custom Vietnamese math metrics
- OpenRouter integration for LLM-as-a-judge evaluation
- Batch processing and streaming support
- Hardware optimization for Tesla T4 and A100

Usage:
    from evaluation_pipeline import EvaluationManager
    from evaluation_pipeline.config import ConfigManager
    
    # Load configuration
    config = ConfigManager.from_yaml("configs/evaluation_config.yaml")
    
    # Run evaluation
    manager = EvaluationManager(config)
    results = manager.evaluate_model("path/to/model")
"""

from .config.config_manager import ConfigManager
from .managers.evaluation_manager import EvaluationManager
from .managers.dataset_manager import DatasetManager
from .managers.metrics_manager import MetricsManager
from .factories.model_factory import ModelFactory
from .factories.evaluator_factory import EvaluatorFactory
from .factories.provider_factory import ProviderFactory

__version__ = "0.1.0"
__author__ = "MathPal Team"

__all__ = [
    "ConfigManager",
    "EvaluationManager", 
    "DatasetManager",
    "MetricsManager",
    "ModelFactory",
    "EvaluatorFactory",
    "ProviderFactory",
]
