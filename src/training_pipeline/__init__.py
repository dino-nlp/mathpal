"""Gemma3N fine-tuning pipeline."""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Modular pipeline for fine-tuning Gemma3N models with Unsloth, TRL, and Comet ML"

from . import config
from . import data
from . import models
from . import training
from . import experiments
from . import inference
from . import utils

__all__ = [
    "config",
    "data", 
    "models",
    "training",
    "experiments",
    "inference",
    "utils"
]