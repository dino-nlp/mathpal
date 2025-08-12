"""Main training orchestrator that coordinates all components."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch

from training_pipeline.utils.exceptions import TrainingError, ModelError, DatasetError
from training_pipeline.config.config_manager import ConfigManager
from training_pipeline.factories import ModelFactory, DatasetFactory, TrainerFactory
from training_pipeline.managers import ExperimentManager, CheckpointManager, EvaluationManager
from training_pipeline.utils import get_logger, DeviceUtils

logger = get_logger()


@dataclass
class TrainingResults:
    """Container for training results."""
    final_loss: float
    training_time: float
    total_steps: int
    model_paths: Dict[str, str]
    evaluation_results: Optional[Dict[str, Any]] = None
    trainer_stats: Optional[Dict[str, Any]] = None


class TrainingManager:
    """
    Main training orchestrator that coordinates all training pipeline components.
    
    This class implements the main training workflow:
    1. Setup experiment tracking
    2. Load model and tokenizer
    3. Prepare dataset
    4. Create trainer
    5. Run training
    6. Save model
    7. Run evaluation (if enabled)
    8. Cleanup
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize training manager.
        
        Args:
            config_manager: Centralized configuration manager
        """
        self.config_manager = config_manager
        
        # Initialize components with specific config sections
        self.model_factory = ModelFactory()
        self.dataset_factory = DatasetFactory()
        self.trainer_factory = TrainerFactory()
        self.experiment_manager = ExperimentManager(
            output_config=config_manager.output,
            comet_config=config_manager.comet
        )
        self.checkpoint_manager = CheckpointManager(
            output_config=config_manager.output,
            hub_config=config_manager.raw_config.get('hub', {})  # Using raw config for hub
        )
        # self.evaluation_manager = EvaluationManager(config_manager)  # TODO: Refactor later
        
        # State tracking
        self.model = None
        self.tokenizer = None
        self.datasets = None
        self.trainer = None
        self.start_time = None
    
    def run_training(self) -> TrainingResults:
        """
        Execute the complete training pipeline.
        
        Returns:
            TrainingResults containing training outcomes
            
        Raises:
            TrainingError: If training fails at any stage
        """
        try:
            logger.info("🚀 Starting MathPal training pipeline...")
            self.start_time = time.time()
            
            # Step 1: Setup experiment tracking
            self._setup_experiment()
            
            # Step 2: Setup environment and validate system
            self._setup_environment()
            
            # Step 3: Load model and tokenizer
            self._load_model()
            
            # Step 4: Prepare dataset
            self._prepare_dataset()
            
            # Step 5: Create trainer
            self._create_trainer()
            
            # Step 6: Run training
            trainer_stats = self._run_training()
            
            # Step 7: Save model
            model_paths = self._save_model(trainer_stats)
            
            # Step 8: Run evaluation (if enabled)
            evaluation_results = self._run_evaluation()
            
            # Step 9: Push to Hub (if enabled)
            self._push_to_hub()
            
            # Calculate final results
            training_time = time.time() - self.start_time
            results = TrainingResults(
                final_loss=self._extract_final_loss(trainer_stats),
                training_time=training_time,
                total_steps=self._extract_total_steps(trainer_stats),
                model_paths=model_paths,
                evaluation_results=evaluation_results,
                trainer_stats=trainer_stats
            )
            
            # Log summary
            self._log_training_summary(results)
            
            return results
            
        except Exception as e:
            self._handle_training_error(e)
            raise
        finally:
            self._cleanup()
    
    def _setup_experiment(self) -> None:
        """Setup experiment tracking and logging."""
        try:
            logger.info("🧪 Setting up experiment tracking...")
            self.experiment_manager.setup()
        except Exception as e:
            raise TrainingError(f"Failed to setup experiment: {e}")
    
    def _setup_environment(self) -> None:
        """Setup training environment and validate system."""
        try:
            logger.info("🔧 Setting up training environment...")
            
            # Print device information
            DeviceUtils.print_device_info()
            
            # Validate memory requirements
            estimated_memory = self.model_factory.estimate_memory_usage(self.config_manager)
            available_memory = DeviceUtils.get_gpu_memory_gb()
            
            if available_memory > 0 and estimated_memory > available_memory * 0.9:
                logger.warning(
                    f"⚠️ High memory usage predicted: {estimated_memory:.1f}GB "
                    f"(Available: {available_memory:.1f}GB)"
                )
                logger.warning("Consider reducing batch size or enabling quantization")
            
            # Set random seed
            from training_pipeline.training.training_utils import TrainingUtils
            TrainingUtils.set_seed(self.config_manager.system.seed)
            
            # Setup output directory
            from training_pipeline.training.training_utils import TrainingUtils
            TrainingUtils.setup_output_directory(self.config_manager.get_output_dir())
            
        except Exception as e:
            raise TrainingError(f"Failed to setup environment: {e}")
    
    def _load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            logger.info("📂 Loading model and tokenizer...")
            logger.info(f"TTTT: {self.config_manager}")
            self.model, self.tokenizer = self.model_factory.create_model(self.config_manager)
            
            # Log model info to experiment tracker
            if self.experiment_manager.is_active():
                self.experiment_manager.log_metrics({
                    "model_parameters": self._count_parameters(self.model),
                    "trainable_parameters": self._count_trainable_parameters(self.model),
                })
                
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")
    
    def _prepare_dataset(self) -> None:
        """Load and prepare datasets."""
        try:
            logger.info("📊 Preparing datasets...")
            self.datasets = self.dataset_factory.create_dataset(self.config_manager, self.tokenizer)
            
            # Log dataset info
            train_size = len(self.datasets["train"])
            eval_size = len(self.datasets.get("eval", []))
            
            if self.experiment_manager.is_active():
                self.experiment_manager.log_metrics({
                    "train_dataset_size": train_size,
                    "eval_dataset_size": eval_size,
                })
                
        except Exception as e:
            raise DatasetError(f"Failed to prepare dataset: {e}")
    
    def _create_trainer(self) -> None:
        """Create and configure trainer."""
        try:
            logger.info("🏋️ Creating trainer...")
            self.trainer = self.trainer_factory.create_trainer(
                self.config_manager, self.model, self.tokenizer, self.datasets
            )
        except Exception as e:
            raise TrainingError(f"Failed to create trainer: {e}")
    
    def _run_training(self) -> Dict[str, Any]:
        """Execute the training process."""
        try:
            logger.info("🚀 Starting training...")
            
            # Start training
            trainer_stats = self.trainer.train()
            
            # Log training completion
            training_time = time.time() - self.start_time
            logger.info(f"✅ Training completed in {training_time:.2f} seconds")
            
            # Log final metrics to experiment tracker
            if self.experiment_manager.is_active():
                self.experiment_manager.log_training_summary(trainer_stats.__dict__)
            
            return trainer_stats.__dict__ if hasattr(trainer_stats, '__dict__') else {}
            
        except Exception as e:
            raise TrainingError(f"Training failed: {e}")
    
    def _save_model(self, trainer_stats: Dict[str, Any]) -> Dict[str, str]:
        """Save trained model in requested formats."""
        try:
            logger.info("💾 Saving trained model...")
            return self.checkpoint_manager.save_model(
                self.model, self.tokenizer, trainer_stats
            )
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return {"error": str(e)}
    
    def _run_evaluation(self) -> Optional[Dict[str, Any]]:
        """Run model evaluation if enabled."""
        try:
            if not self.config_manager.inference.test_after_training:
                return None
            
            logger.info("🧪 Running model evaluation...")
            return self.evaluation_manager.run_evaluation(self.model, self.tokenizer)
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {"error": str(e)}
    
    def _push_to_hub(self) -> Optional[str]:
        """Push model to HuggingFace Hub if enabled."""
        try:
            if not self.config_manager.hub.push_to_hub:
                return None
            
            return self.checkpoint_manager.push_to_hub(self.model, self.tokenizer)
            
        except Exception as e:
            logger.error(f"❌ Failed to push to Hub: {e}")
            return None
    
    def _log_training_summary(self, results: TrainingResults) -> None:
        """Log comprehensive training summary."""
        logger.info("🎉 Training Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info("📊 Training Summary:")
        logger.info(f"   ⏱️  Total time: {results.training_time:.2f} seconds")
        logger.info(f"   📈 Total steps: {results.total_steps:,}")
        logger.info(f"   📉 Final loss: {results.final_loss:.4f}")
        logger.info(f"   💾 Model formats saved: {len(results.model_paths)}")
        
        for format_name, path in results.model_paths.items():
            if not path.startswith("Error"):
                logger.info(f"      📦 {format_name}: {path}")
        
        if results.evaluation_results:
            logger.info(f"   🧪 Evaluation completed: {len(results.evaluation_results)} test suites")
        
        logger.info("=" * 60)
    
    def _handle_training_error(self, error: Exception) -> None:
        """Handle training errors with proper logging."""
        logger.error(f"❌ Training pipeline failed: {error}")
        
        # Log error to experiment tracker
        if self.experiment_manager.is_active():
            self.experiment_manager.log_error(error)
        
        # Clear CUDA cache if available
        DeviceUtils.clear_cuda_cache()
    
    def _cleanup(self) -> None:
        """Cleanup resources and end experiment tracking."""
        try:
            logger.info("🧹 Cleaning up...")
            
            # End experiment tracking
            self.experiment_manager.cleanup()
            
            # Cleanup old checkpoints
            self.checkpoint_manager.cleanup_old_checkpoints()
            
            # Clear CUDA cache
            DeviceUtils.clear_cuda_cache()
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def _extract_final_loss(self, trainer_stats: Dict[str, Any]) -> float:
        """Extract final loss from trainer statistics."""
        try:
            # Try different possible keys for final loss
            for key in ["train_loss", "final_loss", "loss"]:
                if key in trainer_stats:
                    return float(trainer_stats[key])
            return 0.0
        except:
            return 0.0
    
    def _extract_total_steps(self, trainer_stats: Dict[str, Any]) -> int:
        """Extract total steps from trainer statistics."""
        try:
            for key in ["global_step", "step", "steps"]:
                if key in trainer_stats:
                    return int(trainer_stats[key])
            return self.config_manager.training.max_steps
        except:
            return self.config_manager.training.max_steps
    
    def _count_parameters(self, model: Any) -> int:
        """Count total model parameters."""
        try:
            return sum(p.numel() for p in model.parameters())
        except:
            return 0
    
    def _count_trainable_parameters(self, model: Any) -> int:
        """Count trainable model parameters."""
        try:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            return 0
    
    # Additional utility methods for advanced features
    
    def validate_config(self) -> None:
        """Validate configuration before training."""
        try:
            self.config_manager.validate()
            logger.info("✅ Configuration validation passed")
        except Exception as e:
            raise TrainingError(f"Configuration validation failed: {e}")
    
    def estimate_training_cost(self) -> Dict[str, Any]:
        """Estimate training costs and requirements."""
        try:
            estimated_memory = self.model_factory.estimate_memory_usage(self.config_manager)
            available_memory = DeviceUtils.get_gpu_memory_gb()
            
            # Estimate training time
            estimated_time_hours = (self.config_manager.training.max_steps * 1.0) / 3600  # Rough estimate
            
            return {
                "estimated_memory_gb": estimated_memory,
                "available_memory_gb": available_memory,
                "memory_utilization": estimated_memory / available_memory if available_memory > 0 else 0,
                "estimated_time_hours": estimated_time_hours,
                "feasible": estimated_memory <= available_memory * 0.9 if available_memory > 0 else True
            }
        except Exception as e:
            logger.warning(f"Cost estimation failed: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "config_loaded": self.config_manager is not None,
            "model_loaded": self.model is not None,
            "dataset_prepared": self.datasets is not None,
            "trainer_created": self.trainer is not None,
            "training_started": self.start_time is not None,
            "experiment_active": self.experiment_manager.is_active() if self.experiment_manager else False
        }
