"""Command line interface for Gemma3N training pipeline."""

from .train_gemma_v2 import main as train_main_v2

__all__ = [
    "train_main_v2"
]