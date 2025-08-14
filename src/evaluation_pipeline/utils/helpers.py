"""
Helper functions for the evaluation pipeline.
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary containing device information
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "current_device": "cpu",
        "device_name": "CPU",
        "memory_info": {}
    }
    
    if torch.cuda.is_available():
        device_info["device_count"] = torch.cuda.device_count()
        device_info["current_device"] = f"cuda:{torch.cuda.current_device()}"
        device_info["device_name"] = torch.cuda.get_device_name(0)
        
        # Memory info
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            memory_info[f"cuda:{i}"] = {
                "total": torch.cuda.get_device_properties(i).total_memory,
                "allocated": torch.cuda.memory_allocated(i),
                "cached": torch.cuda.memory_reserved(i)
            }
        device_info["memory_info"] = memory_info
    
    return device_info


def validate_model_path(model_path: Union[str, Path]) -> Path:
    """
    Validate that the model path exists and contains required files.
    
    Args:
        model_path: Path to the model
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If model path is invalid
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValidationError(f"Model path does not exist: {model_path}")
    
    # Check for common model files
    required_files = ["config.json", "pytorch_model.bin"]
    optional_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    
    missing_required = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_required.append(file)
    
    if missing_required:
        raise ValidationError(f"Missing required model files: {missing_required}")
    
    return model_path


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If config file cannot be loaded
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Config file does not exist: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML config: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config file: {e}")


def save_yaml_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
        
    Raises:
        ConfigurationError: If config cannot be saved
    """
    config_path = Path(config_path)
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise ConfigurationError(f"Error saving config file: {e}")


def format_memory_size(bytes_size: int) -> str:
    """
    Format memory size in human readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_environment_variables() -> Dict[str, str]:
    """
    Get relevant environment variables for the evaluation pipeline.
    
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
    # API keys
    for key in ['OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
        if key in os.environ:
            env_vars[key] = os.environ[key][:8] + "..."  # Truncate for security
    
    # Other relevant variables
    for key in ['CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PWD']:
        if key in os.environ:
            env_vars[key] = os.environ[key]
    
    return env_vars


def create_output_directory(output_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create output directory for evaluation results.
    
    Args:
        output_dir: Base output directory
        experiment_name: Name of the experiment
        
    Returns:
        Path to created directory
    """
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    filename: str = "evaluation_results.json"
) -> Path:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output directory path
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        return file_path
    except Exception as e:
        raise ConfigurationError(f"Error saving evaluation results: {e}")


def load_evaluation_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Evaluation results dictionary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ConfigurationError(f"Results file does not exist: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ConfigurationError(f"Error loading evaluation results: {e}")


def validate_metrics_config(metrics_config: List[str]) -> List[str]:
    """
    Validate metrics configuration.
    
    Args:
        metrics_config: List of metric names
        
    Returns:
        Validated list of metrics
        
    Raises:
        ValidationError: If metrics are invalid
    """
    valid_metrics = {
        # Opik built-in metrics
        "hallucination", "context_precision", "context_recall",
        "answer_relevance", "usefulness", "moderation",
        "conversational_coherence", "session_completeness_quality",
        "user_frustration",
        
        # Custom Vietnamese math metrics
        "mathematical_accuracy", "vietnamese_language_quality",
        "step_by_step_reasoning", "grade_level_appropriateness",
        "problem_solving_approach"
    }
    
    invalid_metrics = []
    for metric in metrics_config:
        if metric not in valid_metrics:
            invalid_metrics.append(metric)
    
    if invalid_metrics:
        raise ValidationError(f"Invalid metrics: {invalid_metrics}")
    
    return metrics_config


def get_model_size_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about model size and files.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary with model size information
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValidationError(f"Model path does not exist: {model_path}")
    
    total_size = 0
    file_info = {}
    
    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            total_size += file_size
            file_info[str(file_path.relative_to(model_path))] = {
                "size": file_size,
                "size_formatted": format_memory_size(file_size)
            }
    
    return {
        "total_size": total_size,
        "total_size_formatted": format_memory_size(total_size),
        "file_count": len(file_info),
        "files": file_info
    }
