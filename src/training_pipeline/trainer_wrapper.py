"""
Training wrapper with Comet ML integration for Gemma3N fine-tuning
Provides high-level training interface with comprehensive monitoring
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import torch

from transformers import TrainingArguments, EarlyStoppingCallback
from transformers.utils import logging as transformers_logging
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Comet ML imports
import comet_ml

from config import ExperimentConfig, TrainingConfig, CometConfig
from model_manager import GemmaModelManager
from data_processor import MathDatasetProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CometMLIntegration:
    """
    Comet ML integration cho experiment tracking và model registry
    """
    
    def __init__(self, config: CometConfig):
        self.config = config
        self.experiment = None
        self.model_logged = False
        
    def setup_experiment(self, training_config: TrainingConfig) -> Optional[comet_ml.Experiment]:
        """
        Setup Comet ML experiment với full configuration
        """
        try:
            logger.info("🔧 Setting up Comet ML experiment...")
            
            # Validate configuration
            if not self.config.api_key:
                logger.error("COMET_API_KEY not set")
                return None
            
            # Initialize experiment
            experiment_kwargs = {
                "api_key": self.config.api_key,
                "workspace": self.config.workspace,
                "project_name": self.config.project,
                "experiment_name": self.config.experiment_name,
                "auto_metric_logging": self.config.auto_metric_logging,
                "auto_param_logging": self.config.auto_param_logging,
                "auto_histogram_weight_logging": self.config.auto_histogram_weight_logging,
                "auto_histogram_gradient_logging": self.config.auto_histogram_gradient_logging,
                "auto_histogram_activation_logging": self.config.auto_histogram_activation_logging,
            }
            
            # Remove None values
            experiment_kwargs = {k: v for k, v in experiment_kwargs.items() if v is not None}
            
            self.experiment = comet_ml.Experiment(**experiment_kwargs)
            
            # Add tags
            for tag in self.config.tags:
                self.experiment.add_tag(tag)
            
            # Log training configuration
            self._log_training_config(training_config)
            
            # Set environment variables cho transformers integration
            os.environ["COMET_PROJECT_NAME"] = self.config.project
            if self.config.workspace:
                os.environ["COMET_WORKSPACE"] = self.config.workspace
            
            logger.info(f"✅ Comet ML experiment initialized: {self.experiment.url}")
            
            return self.experiment
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Comet ML: {e}")
            return None
    
    def _log_training_config(self, training_config: TrainingConfig):
        """Log training configuration to Comet"""
        if not self.experiment:
            return
            
        try:
            # Log hyperparameters
            params = {
                "learning_rate": training_config.learning_rate,
                "batch_size": training_config.per_device_train_batch_size,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "num_epochs": training_config.num_train_epochs,
                "warmup_ratio": training_config.warmup_ratio,
                "weight_decay": training_config.weight_decay,
                "optimizer": training_config.optim,
                "lr_scheduler": training_config.lr_scheduler_type,
                "fp16": training_config.fp16,
                "bf16": training_config.bf16,
                "effective_batch_size": (
                    training_config.per_device_train_batch_size * 
                    training_config.gradient_accumulation_steps
                ),
            }
            
            for key, value in params.items():
                self.experiment.log_parameter(key, value)
            
            # Log other metadata
            self.experiment.log_other("framework", "unsloth")
            self.experiment.log_other("task", "math-tutoring")
            self.experiment.log_other("language", "vietnamese")
            self.experiment.log_other("model_type", "gemma3n")
            
        except Exception as e:
            logger.warning(f"Failed to log training config: {e}")
    
    def log_model_to_registry(
        self, 
        model_path: str, 
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log model to Comet ML Model Registry
        """
        if not self.experiment or self.model_logged:
            return
            
        try:
            logger.info("📦 Logging model to Comet ML Model Registry...")
            
            model_name = model_name or self.config.model_registry_name
            
            # Create model registry entry
            self.experiment.log_model(
                name=model_name,
                file_or_folder=model_path,
                metadata=metadata or {}
            )
            
            self.model_logged = True
            logger.info(f"✅ Model logged to registry: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log model to registry: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        if not self.experiment:
            return
            
        try:
            for key, value in dataset_info.items():
                self.experiment.log_parameter(f"dataset_{key}", value)
                
        except Exception as e:
            logger.warning(f"Failed to log dataset info: {e}")
    
    def end_experiment(self):
        """End the experiment"""
        if self.experiment:
            self.experiment.end()
            logger.info("🏁 Comet ML experiment ended")


class TrainingWrapper:
    """
    High-level wrapper cho training process với Comet ML integration
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_manager = None
        self.data_processor = None
        self.trainer = None
        self.comet_integration = None
        
        # Training state
        self.training_completed = False
        self.best_checkpoint = None
        self.training_stats = None
        
    def setup_components(self):
        """Setup tất cả components cần thiết"""
        logger.info("🔧 Setting up training components...")
        
        # Setup Comet ML
        self.comet_integration = CometMLIntegration(self.config.comet)
        experiment = self.comet_integration.setup_experiment(self.config.training)
        
        # Setup model manager
        from model_manager import create_model_manager
        self.model_manager = create_model_manager(self.config.model)
        
        # Setup data processor
        from data_processor import create_data_processor
        self.data_processor = create_data_processor(self.config.dataset)
        
        logger.info("✅ All components setup successfully")
        
        return experiment
    
    def prepare_data(self) -> Dict[str, Dataset]:
        """Prepare training and evaluation datasets"""
        logger.info("📚 Preparing datasets...")
        
        # Load and process datasets
        self.data_processor.load_dataset()
        
        # Set tokenizer after model is loaded
        if self.model_manager.tokenizer is None:
            raise ValueError("Model manager tokenizer not available")
        
        self.data_processor.set_tokenizer(self.model_manager.tokenizer)
        
        # Prepare datasets
        datasets = self.data_processor.prepare_datasets()
        
        # Log dataset info to Comet
        if self.comet_integration.experiment:
            dataset_info = {
                "name": self.config.dataset.dataset_name,
                "train_samples": len(datasets.get("train", [])),
                "eval_samples": len(datasets.get("eval", [])),
                "max_seq_length": self.config.model.max_seq_length,
            }
            self.comet_integration.log_dataset_info(dataset_info)
        
        logger.info("✅ Datasets prepared successfully")
        
        return datasets
    
    def create_trainer(self, datasets: Dict[str, Dataset]) -> SFTTrainer:
        """Create SFTTrainer với optimized configuration"""
        logger.info("🏋️ Creating trainer...")
        
        # Prepare model for training theo working notebook pattern
        from unsloth import FastModel
        FastModel.for_training(self.model_manager.model)
        
        # Create SFTConfig theo working notebook pattern (minimal parameters)
        training_args = SFTConfig(
            # Dataset settings - theo working pattern
            dataset_text_field=self.config.dataset.dataset_text_field,
            
            # Basic settings
            output_dir=self.config.training.output_dir,
            max_steps=self.config.training.max_steps,
            
            # Batch settings
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            
            # Learning rate
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            
            # Optimization - theo working notebook
            optim=self.config.training.optim,
            
            # Logging and saving
            logging_steps=self.config.training.logging_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            report_to=self.config.training.report_to,
            
            # Length settings - key difference: max_length không phải max_seq_length
            max_length=self.config.model.max_seq_length,
            
            # Reproducibility
            seed=self.config.training.seed,
        )
        
        # Create trainer theo working notebook pattern (minimal parameters)
        trainer = SFTTrainer(
            model=self.model_manager.model,
            tokenizer=self.model_manager.tokenizer,
            train_dataset=datasets["train"],
            # Không include eval_dataset như working notebook
            args=training_args,
        )
        
        # Apply train_on_responses_only để chỉ train trên assistant responses
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        
        # Add early stopping callback
        if hasattr(self.config.training, 'early_stopping_patience'):
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.config.training.early_stopping_patience,
                early_stopping_threshold=getattr(
                    self.config.training, 'early_stopping_threshold', 0.0
                )
            )
            trainer.add_callback(early_stopping_callback)
        
        self.trainer = trainer
        logger.info("✅ Trainer created successfully")
        
        return trainer
    
    def run_training(self) -> Dict[str, Any]:
        """Execute training process"""
        if not self.trainer:
            raise ValueError("Trainer not created. Call create_trainer() first.")
        
        logger.info("🚀 Starting training process...")
        
        # Log memory usage before training
        memory_info = self.model_manager.get_memory_usage()
        logger.info(f"📊 Memory usage before training: {memory_info}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run training
            train_result = self.trainer.train()
            
            # Record training stats
            end_time = time.time()
            self.training_stats = {
                "train_runtime": train_result.metrics.get("train_runtime", end_time - start_time),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
                "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
                "total_flos": train_result.metrics.get("total_flos"),
                "train_loss": train_result.metrics.get("train_loss"),
                "eval_loss": train_result.metrics.get("eval_loss"),
                "best_checkpoint": self.trainer.state.best_model_checkpoint,
            }
            
            # Log final memory usage
            final_memory = self.model_manager.get_memory_usage()
            logger.info(f"📊 Memory usage after training: {final_memory}")
            
            self.training_completed = True
            self.best_checkpoint = self.trainer.state.best_model_checkpoint
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"📈 Training stats: {self.training_stats}")
            
            return self.training_stats
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_model(self, save_formats: List[str] = None) -> Dict[str, str]:
        """
        Save model in multiple formats và log to Comet ML registry
        """
        if not self.training_completed:
            logger.warning("Training not completed, saving current state...")
        
        save_formats = save_formats or ["lora", "merged_16bit"]
        saved_paths = {}
        
        logger.info(f"💾 Saving model in formats: {save_formats}")
        
        for save_format in save_formats:
            try:
                # Create format-specific output directory
                format_dir = Path(self.config.training.output_dir) / f"model_{save_format}"
                
                # Save model
                self.model_manager.save_model(
                    output_dir=str(format_dir),
                    save_method=save_format
                )
                
                saved_paths[save_format] = str(format_dir)
                
                # Log to Comet ML model registry
                if self.comet_integration and save_format == "lora":  # Log LoRA to registry
                    metadata = {
                        "save_format": save_format,
                        "training_completed": self.training_completed,
                        "best_checkpoint": self.best_checkpoint,
                        "training_stats": self.training_stats,
                        "model_config": self.config.model.to_dict() if hasattr(self.config.model, 'to_dict') else str(self.config.model),
                    }
                    
                    self.comet_integration.log_model_to_registry(
                        model_path=str(format_dir),
                        model_name=f"{self.config.comet.model_registry_name}_{save_format}",
                        metadata=metadata
                    )
                
                logger.info(f"✅ Model saved in {save_format} format: {format_dir}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save model in {save_format} format: {e}")
        
        return saved_paths
    
    def push_to_hub(
        self, 
        repo_id: str, 
        save_formats: List[str] = None,
        token: Optional[str] = None,
        private: bool = True
    ):
        """Push model to HuggingFace Hub"""
        save_formats = save_formats or ["lora"]
        
        logger.info(f"☁️  Pushing model to Hub: {repo_id}")
        
        for save_format in save_formats:
            try:
                format_repo_id = f"{repo_id}-{save_format}" if len(save_formats) > 1 else repo_id
                
                self.model_manager.push_to_hub(
                    repo_id=format_repo_id,
                    save_method=save_format,
                    token=token,
                    private=private
                )
                
                logger.info(f"✅ Model pushed to Hub in {save_format} format: {format_repo_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to push model in {save_format} format: {e}")
    
    def run_inference_test(self, test_questions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run inference test với sample questions
        """
        logger.info("🧪 Running inference test...")
        
        # Prepare model for inference
        self.model_manager.prepare_for_inference()
        
        # Get test questions
        if test_questions is None:
            # Use questions from eval dataset
            try:
                sample = self.data_processor.get_sample_for_inference(0)
                test_questions = [sample["question"]]
            except:
                test_questions = ["Tính 5 + 3 = ?", "Một hình chữ nhật có chiều dài 8m và chiều rộng 5m. Tính diện tích?"]
        
        results = []
        
        for i, question in enumerate(test_questions):
            try:
                logger.info(f"Testing question {i+1}: {question[:50]}...")
                
                response = self.model_manager.generate_response(
                    question=question,
                    inference_config=self.config.inference
                )
                
                result = {
                    "question": question,
                    "response": response,
                    "success": True
                }
                
                logger.info(f"✅ Response {i+1}: {response[:100]}...")
                
            except Exception as e:
                result = {
                    "question": question,
                    "response": None,
                    "error": str(e),
                    "success": False
                }
                logger.error(f"❌ Inference failed for question {i+1}: {e}")
            
            results.append(result)
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        # End Comet ML experiment
        if self.comet_integration:
            self.comet_integration.end_experiment()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Cleanup completed")


def create_trainer_wrapper(config: ExperimentConfig) -> TrainingWrapper:
    """
    Factory function to create training wrapper
    """
    return TrainingWrapper(config)


def run_complete_training_pipeline(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run complete training pipeline từ setup đến save model
    """
    logger.info("🎯 Starting complete training pipeline...")
    
    # Create wrapper
    wrapper = create_trainer_wrapper(config)
    
    try:
        # Setup components
        experiment = wrapper.setup_components()
        
        # Load model
        model, tokenizer = wrapper.model_manager.load_base_model()
        model = wrapper.model_manager.apply_lora()
        
        # Prepare data
        datasets = wrapper.prepare_data()
        
        # Create trainer
        trainer = wrapper.create_trainer(datasets)
        
        # Run training
        training_stats = wrapper.run_training()
        
        # Save model
        saved_paths = wrapper.save_model(save_formats=["lora", "merged_16bit"])
        
        # Run inference test
        inference_results = wrapper.run_inference_test()
        
        # Create summary
        pipeline_results = {
            "training_stats": training_stats,
            "saved_paths": saved_paths,
            "inference_results": inference_results,
            "experiment_url": experiment.url if experiment else None,
            "best_checkpoint": wrapper.best_checkpoint,
            "config": config.to_dict()
        }
        
        # Save results
        results_path = Path(config.training.output_dir) / "pipeline_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Complete training pipeline finished successfully!")
        logger.info(f"📊 Results saved to: {results_path}")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise
    finally:
        # Always cleanup
        wrapper.cleanup()


if __name__ == "__main__":
    # Test training wrapper
    from config import get_optimized_config_for_t4
    
    config = get_optimized_config_for_t4()
    config.training.num_train_epochs = 1  # Quick test
    config.dataset.max_samples = 10       # Small dataset for test
    
    try:
        results = run_complete_training_pipeline(config)
        logger.info("✅ Training wrapper test passed!")
        
    except Exception as e:
        logger.error(f"❌ Training wrapper test failed: {e}")
