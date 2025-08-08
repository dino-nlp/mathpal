"""Comet ML configuration."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .base_config import BaseConfig, get_env_var


@dataclass
class CometConfig(BaseConfig):
    """Configuration for Comet ML experiment tracking."""
    
    # Required settings
    api_key: Optional[str] = field(default_factory=lambda: get_env_var("COMET_API_KEY"))
    workspace: Optional[str] = field(default_factory=lambda: get_env_var("COMET_WORKSPACE"))
    project: Optional[str] = field(default_factory=lambda: get_env_var("COMET_PROJECT"))
    
    # Optional settings
    experiment_name: str = "gemma3n_finetune"
    tags: List[str] = field(default_factory=lambda: [
        "gemma3n",
        "math-tutor", 
        "vietnamese",
        "sixth-grade",
        "fine-tuning"
    ])
    
    # Logging settings
    auto_metric_logging: bool = True
    auto_param_logging: bool = True
    auto_histogram_weight_logging: bool = True
    auto_histogram_gradient_logging: bool = True
    auto_histogram_activation_logging: bool = False
    auto_output_logging: str = "default"
    
    # Model logging
    log_model: bool = True
    log_graph: bool = False
    log_code: bool = True
    log_git_metadata: bool = True
    
    def validate(self) -> None:
        """Validate Comet ML configuration."""
        if not self.api_key:
            raise ValueError("COMET_API_KEY environment variable is required")
        if not self.workspace:
            raise ValueError("COMET_WORKSPACE environment variable is required") 
        if not self.project:
            raise ValueError("COMET_PROJECT environment variable is required")
    
    def get_experiment_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for Comet ML experiment initialization."""
        return {
            "workspace": self.workspace,
            "project_name": self.project,
            "auto_metric_logging": self.auto_metric_logging,
            "auto_param_logging": self.auto_param_logging,
            "auto_histogram_weight_logging": self.auto_histogram_weight_logging,
            "auto_histogram_gradient_logging": self.auto_histogram_gradient_logging,
            "auto_histogram_activation_logging": self.auto_histogram_activation_logging,
        }
    
    def setup_environment(self) -> None:
        """Setup environment variables for transformers integration."""
        import os
        if self.project:
            os.environ["COMET_PROJECT_NAME"] = self.project
        if self.workspace:
            os.environ["COMET_WORKSPACE"] = self.workspace