"""Configuration management for Gemma3N fine-tuning pipeline."""

from .config_manager import (
    ConfigManager, create_config_manager,
    ModelConfig, DatasetConfig, TrainingConfig, LoRAConfig, SystemConfig,
    OutputConfig, EvaluationConfig, LoggingConfig, CometConfig, InferenceConfig,
    HubConfig, ComprehensiveTrainingConfig, ConfigLoader,
    ModelConfigSection, DatasetConfigSection, TrainingConfigSection, 
    LoRAConfigSection, OutputConfigSection, CometConfigSection, SystemConfigSection
)

__all__ = [
    # New ConfigManager system
    "ConfigManager", "create_config_manager",
    # Config sections (new)
    "ModelConfigSection", "DatasetConfigSection", "TrainingConfigSection", 
    "LoRAConfigSection", "OutputConfigSection", "CometConfigSection", "SystemConfigSection",
    # Legacy classes (backward compatibility)
    "ModelConfig", "DatasetConfig", "TrainingConfig", "LoRAConfig", "SystemConfig",
    "OutputConfig", "EvaluationConfig", "LoggingConfig", "CometConfig", "InferenceConfig", 
    "HubConfig", "ComprehensiveTrainingConfig", "ConfigLoader"
]