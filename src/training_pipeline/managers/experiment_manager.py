"""Experiment tracking and management."""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

from ..core.exceptions import ExperimentError
from ..config.config_manager import OutputConfigSection, CometConfigSection
from ..utils import get_logger

logger = get_logger()


class ExperimentManager:
    """Manages experiment tracking and monitoring."""
    
    def __init__(self, output_config: OutputConfigSection, comet_config: CometConfigSection):
        """
        Initialize ExperimentManager with specific config sections.
        
        Args:
            output_config: Output configuration section
            comet_config: Comet ML configuration section
        """
        self.output_config = output_config
        self.comet_config = comet_config
        self.comet_experiment = None
        self.start_time = None
        self.experiment_id = None
        
    def setup(self) -> None:
        """Setup experiment tracking."""
        try:
            self.start_time = time.time()
            self.experiment_id = f"{self.output_config.experiment_name}_{int(self.start_time)}"
            
            logger.info(f"🧪 Setting up experiment: {self.experiment_id}")
            
            # Setup output directory
            self._setup_output_directory()
            
            # Setup Comet ML if enabled
            if self.comet_config.enabled:
                self._setup_comet()
            
            # Log experiment info
            self._log_experiment_info()
            
        except Exception as e:
            raise ExperimentError(f"Failed to setup experiment: {e}")
    
    def _setup_output_directory(self) -> None:
        """Create and setup output directory."""
        output_dir = self.output_config.get_output_dir()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Output directory: {output_dir}")
        # Note: Config saving is now handled by ConfigManager, not here
    
    def _setup_comet(self) -> None:
        """Setup Comet ML experiment tracking."""
        try:
            import comet_ml
            
            # Get credentials from environment or config
            api_key = os.getenv("COMET_API_KEY")
            workspace = os.getenv("COMET_WORKSPACE")
            project = os.getenv("COMET_PROJECT", "mathpal-training")
            
            if not api_key:
                logger.warning("⚠️ COMET_API_KEY not found, skipping Comet ML setup")
                return
            
            # Create experiment
            self.comet_experiment = comet_ml.Experiment(
                api_key=api_key,
                workspace=workspace,
                project_name=project,
                auto_metric_logging=self.comet_config.auto_metric_logging,
                auto_param_logging=self.comet_config.auto_param_logging,
            )
            
            # Set experiment name and tags
            self.comet_experiment.set_name(self.comet_config.experiment_name)
            self.comet_experiment.add_tags(self.comet_config.tags)
            
            # Log configuration sections
            config_dict = {
                "output": self.output_config.to_dict(),
                "comet": self.comet_config.to_dict()
            }
            self.comet_experiment.log_parameters(config_dict)
            
            logger.info("✅ Comet ML experiment created")
            
        except ImportError:
            logger.warning("⚠️ Comet ML not installed, skipping setup")
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup Comet ML: {e}")
    
    def _log_experiment_info(self) -> None:
        """Log basic experiment information."""
        logger.info("📊 Experiment Information:")
        logger.info(f"   🆔 ID: {self.experiment_id}")
        logger.info(f"   📛 Name: {self.output_config.experiment_name}")
        logger.info(f"   📁 Output: {self.output_config.get_output_dir()}")
        logger.info(f"   📊 Comet enabled: {self.comet_config.enabled}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to tracking systems."""
        try:
            if self.comet_experiment:
                self.comet_experiment.log_metric(name, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metric {name}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_training_summary(self, trainer_stats: Dict[str, Any]) -> None:
        """Log training summary statistics."""
        try:
            if self.comet_experiment and trainer_stats:
                # Log final metrics
                for key, value in trainer_stats.items():
                    if isinstance(value, (int, float)):
                        self.comet_experiment.log_metric(f"final_{key}", value)
                
                # Log training time
                if self.start_time:
                    training_time = time.time() - self.start_time
                    self.comet_experiment.log_metric("training_time_seconds", training_time)
                    
                logger.info("📊 Training summary logged to Comet ML")
                
        except Exception as e:
            logger.warning(f"Failed to log training summary: {e}")
    
    def log_model(self, model_path: str, model_name: str) -> None:
        """Log model artifacts."""
        try:
            if self.comet_experiment and os.path.exists(model_path):
                self.comet_experiment.log_model(model_name, model_path)
                logger.info(f"📦 Model logged to Comet ML: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to log model {model_name}: {e}")
    
    def log_error(self, error: Exception) -> None:
        """Log error information."""
        try:
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "experiment_id": self.experiment_id
            }
            
            if self.comet_experiment:
                self.comet_experiment.log_other("error", error_info)
                
            logger.error(f"❌ Error logged: {error_info}")
            
        except Exception as e:
            logger.warning(f"Failed to log error: {e}")
    
    def end_experiment(self) -> None:
        """End experiment tracking."""
        try:
            if self.comet_experiment:
                self.comet_experiment.end()
                logger.info("🔚 Comet ML experiment ended")
                
            if self.start_time:
                total_time = time.time() - self.start_time
                logger.info(f"⏱️ Total experiment time: {total_time:.2f} seconds")
                
        except Exception as e:
            logger.warning(f"Failed to end experiment: {e}")
    
    def cleanup(self) -> None:
        """Cleanup experiment resources."""
        try:
            self.end_experiment()
        except Exception as e:
            logger.warning(f"Failed to cleanup experiment: {e}")
    
    def is_active(self) -> bool:
        """Check if experiment tracking is active."""
        return self.comet_experiment is not None
