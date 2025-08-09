"""Comet ML experiment tracking."""

import os
from typing import Dict, Any, Optional, List
from ..config.comet_config import CometConfig


class CometTracker:
    """Handles Comet ML experiment tracking integration."""
    
    def __init__(self, config: CometConfig):
        """Initialize CometTracker with configuration."""
        self.config = config
        self.experiment = None
        
    def setup_experiment(
        self,
        training_config: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Setup Comet ML experiment with full configuration support.
        
        Args:
            training_config: Training configuration dictionary
            model_config: Optional model configuration
            
        Returns:
            Comet experiment object or None if setup fails
        """
        if self.config.api_key is None:
            print("❌ Comet API key not found. Skipping experiment tracking.")
            return None
            
        try:
            import comet_ml
            
            # Validate configuration
            self.config.validate()
            
            # Get experiment kwargs
            experiment_kwargs = self.config.get_experiment_kwargs()
            
            # Remove None values
            experiment_kwargs = {
                k: v for k, v in experiment_kwargs.items() 
                if v is not None
            }
            
            # Initialize experiment
            self.experiment = comet_ml.Experiment(**experiment_kwargs)
            
            # Log configurations
            self._log_configurations(training_config, model_config)
            
            # Add tags
            self._add_tags()
            
            # Log additional metadata
            self._log_metadata(training_config)
            
            # Setup environment variables for transformers integration
            self.config.setup_environment()
            
            self._print_experiment_info()
            
            return self.experiment
            
        except ImportError:
            print("❌ comet_ml not installed. Please install with: pip install comet-ml")
            print("Falling back to local logging...")
            return None
        except Exception as e:
            print(f"❌ Failed to initialize Comet ML: {e}")
            print("Possible causes:")
            print("- Invalid API key or workspace/project names")
            print("- Network connection issues")
            print("- Missing permissions")
            print("Falling back to local logging...")
            return None
    
    def _log_configurations(
        self,
        training_config: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log configuration parameters."""
        if self.experiment is None:
            return
            
        # Log training configuration
        self.experiment.log_parameters(training_config)
        
        # Log model configuration if provided
        if model_config:
            self.experiment.log_parameters(model_config, prefix="model_")
    
    def _add_tags(self) -> None:
        """Add tags to the experiment."""
        if self.experiment is None:
            return
            
        for tag in self.config.tags:
            self.experiment.add_tag(tag)
    
    def _log_metadata(self, training_config: Dict[str, Any]) -> None:
        """Log additional metadata."""
        if self.experiment is None:
            return
            
        # Log dataset information
        if "dataset_name" in training_config:
            self.experiment.log_other("dataset", training_config["dataset_name"])
        
        if "model_name" in training_config:
            self.experiment.log_other("model_base", training_config["model_name"])
        
        # Log task and language information
        self.experiment.log_other("task", "sixth-grade-math-tutoring")
        self.experiment.log_other("language", "vietnamese")
        self.experiment.log_other("framework", "unsloth+trl")
    
    def _print_experiment_info(self) -> None:
        """Print experiment information."""
        if self.experiment is None:
            return
            
        print("✅ Comet ML experiment initialized")
        print(f"🔗 Experiment URL: {self.experiment.url}")
        print(f"📊 Workspace: {self.config.workspace}")
        print(f"📁 Project: {self.config.project}")
        print(f"🏷️ Tags: {', '.join(self.config.tags)}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric to Comet.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.experiment is None:
            return
            
        try:
            self.experiment.log_metric(name, value, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metric {name}: {e}")
    
    def log_parameter(self, name: str, value: Any) -> None:
        """
        Log a parameter to Comet.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        if self.experiment is None:
            return
            
        try:
            self.experiment.log_parameter(name, value)
        except Exception as e:
            print(f"Warning: Failed to log parameter {name}: {e}")
    
    def log_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        overwrite: bool = False
    ) -> None:
        """
        Log model to Comet.
        
        Args:
            model_path: Path to model directory or file
            model_name: Optional model name
            overwrite: Whether to overwrite existing model
        """
        if self.experiment is None:
            return
            
        try:
            self.experiment.log_model(
                name=model_name or "trained_model",
                file_or_folder=model_path,
                overwrite=overwrite
            )
            print(f"📦 Model logged to Comet: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to log model: {e}")
    
    def log_dataset(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None
    ) -> None:
        """
        Log dataset to Comet.
        
        Args:
            dataset_path: Path to dataset
            dataset_name: Optional dataset name
        """
        if self.experiment is None:
            return
            
        try:
            self.experiment.log_dataset_info(
                name=dataset_name or "training_dataset",
                path=dataset_path
            )
            print(f"📊 Dataset logged to Comet: {dataset_path}")
        except Exception as e:
            print(f"Warning: Failed to log dataset: {e}")
    
    def log_code(self, code_path: str = ".") -> None:
        """
        Log code to Comet.
        
        Args:
            code_path: Path to code directory
        """
        if self.experiment is None or not self.config.log_code:
            return
            
        try:
            self.experiment.log_code(code_path)
            print(f"💻 Code logged to Comet: {code_path}")
        except Exception as e:
            print(f"Warning: Failed to log code: {e}")
    
    def log_training_summary(
        self,
        training_stats: Dict[str, Any],
        final_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log training summary.
        
        Args:
            training_stats: Training statistics
            final_metrics: Optional final metrics
        """
        if self.experiment is None:
            return
            
        try:
            # Log training stats
            for key, value in training_stats.items():
                if isinstance(value, (int, float)):
                    self.log_metric(f"final_{key}", value)
                else:
                    self.log_parameter(f"final_{key}", str(value))
            
            # Log final metrics
            if final_metrics:
                for key, value in final_metrics.items():
                    self.log_metric(f"final_{key}", value)
                    
            print("📈 Training summary logged to Comet")
            
        except Exception as e:
            print(f"Warning: Failed to log training summary: {e}")
    
    def end_experiment(self) -> None:
        """End the Comet experiment."""
        if self.experiment is None:
            return
            
        try:
            self.experiment.end()
            print("🏁 Comet experiment ended")
        except Exception as e:
            print(f"Warning: Failed to end experiment: {e}")
    
    def is_active(self) -> bool:
        """Check if experiment is active."""
        return self.experiment is not None
    
    @staticmethod
    def create_from_env() -> "CometTracker":
        """
        Create CometTracker from environment variables.
        
        Returns:
            CometTracker instance
        """
        config = CometConfig()
        return CometTracker(config)
    
    def get_experiment_url(self) -> Optional[str]:
        """
        Get experiment URL.
        
        Returns:
            Experiment URL or None
        """
        if self.experiment is None:
            return None
        return getattr(self.experiment, 'url', None)
    
    def get_experiment_key(self) -> Optional[str]:
        """
        Get experiment key.
        
        Returns:
            Experiment key or None
        """
        if self.experiment is None:
            return None
        return getattr(self.experiment, 'id', None)