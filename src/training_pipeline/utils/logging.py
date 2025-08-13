"""Logging utilities for the training pipeline."""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger only if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=numeric_level,
            format=format_string,
            handlers=[]
        )
    
    # Set transformers logger level
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    # Create logger
    logger = logging.getLogger("mathpal_training")
    logger.setLevel(numeric_level)
    
    # Prevent propagation to root logger to avoid duplicates
    logger.propagate = False
    
    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: str = "mathpal_training") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """Custom logger for training pipeline with enhanced formatting."""
    
    def __init__(self, logger_name: str = "mathpal_training"):
        """Initialize TrainingLogger."""
        self.logger = get_logger(logger_name)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with emoji."""
        self.logger.info(f"ℹ️ {message}", **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with emoji."""
        self.logger.warning(f"⚠️ {message}", **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with emoji."""
        self.logger.error(f"❌ {message}", **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Log success message with emoji."""
        self.logger.info(f"✅ {message}", **kwargs)
    
    def step(self, step: int, message: str, **kwargs) -> None:
        """Log step message."""
        self.logger.info(f"📍 Step {step}: {message}", **kwargs)
    
    def metric(self, name: str, value: float, step: Optional[int] = None, **kwargs) -> None:
        """Log metric value."""
        step_info = f" (step {step})" if step is not None else ""
        self.logger.info(f"📊 {name}: {value:.4f}{step_info}", **kwargs)
    
    def config(self, config_name: str, config_dict: dict, **kwargs) -> None:
        """Log configuration."""
        self.logger.info(f"🔧 {config_name} Configuration:", **kwargs)
        for key, value in config_dict.items():
            self.logger.info(f"   {key}: {value}", **kwargs)
    
    def progress(self, current: int, total: int, message: str = "", **kwargs) -> None:
        """Log progress."""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = f"🔄 Progress: {current}/{total} ({percentage:.1f}%)"
        if message:
            progress_msg += f" - {message}"
        self.logger.info(progress_msg, **kwargs)
    
    def separator(self, title: str = "", **kwargs) -> None:
        """Log separator line."""
        if title:
            self.logger.info(f"\n{'=' * 20} {title} {'=' * 20}", **kwargs)
        else:
            self.logger.info("=" * 50, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(f"🐛 {message}", **kwargs)
    
    def model_info(self, model_name: str, param_count: int, trainable_count: int, **kwargs) -> None:
        """Log model information."""
        trainable_percent = (trainable_count / param_count * 100) if param_count > 0 else 0
        self.logger.info(f"🤖 Model: {model_name}", **kwargs)
        self.logger.info(f"   Total parameters: {param_count:,}", **kwargs)
        self.logger.info(f"   Trainable parameters: {trainable_count:,}", **kwargs)
        self.logger.info(f"   Trainable%: {trainable_percent:.2f}%", **kwargs)
    
    def dataset_info(self, dataset_name: str, train_size: int, eval_size: Optional[int] = None, **kwargs) -> None:
        """Log dataset information."""
        self.logger.info(f"📊 Dataset: {dataset_name}", **kwargs)
        self.logger.info(f"   Training samples: {train_size:,}", **kwargs)
        if eval_size is not None:
            self.logger.info(f"   Evaluation samples: {eval_size:,}", **kwargs)
    
    def training_start(self, **kwargs) -> None:
        """Log training start."""
        self.separator("TRAINING STARTED")
        self.logger.info("🚀 Starting model training...", **kwargs)
    
    def training_complete(self, duration: float, **kwargs) -> None:
        """Log training completion."""
        self.logger.info(f"🏁 Training completed in {duration:.2f} seconds", **kwargs)
        self.separator("TRAINING COMPLETED")
    
    def save_model(self, path: str, format_type: str = "default", **kwargs) -> None:
        """Log model saving."""
        self.logger.info(f"💾 Saving model ({format_type}) to: {path}", **kwargs)
    
    def load_model(self, path: str, **kwargs) -> None:
        """Log model loading."""
        self.logger.info(f"📂 Loading model from: {path}", **kwargs)