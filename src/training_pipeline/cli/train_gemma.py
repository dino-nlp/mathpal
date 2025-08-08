"""Main training script for Gemma3N fine-tuning."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training_pipeline.config import TrainingConfig, CometConfig
from training_pipeline.data import DatasetProcessor
from training_pipeline.models import ModelLoader, ModelSaver
from training_pipeline.training import TrainerFactory, TrainingUtils
from training_pipeline.experiments import CometTracker
from training_pipeline.inference import InferenceEngine
from training_pipeline.utils import setup_logging, get_logger, DeviceUtils


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma3N model for Vietnamese math tutoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    
    # Model settings
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-3n-E4B-it",
        help="Model name or path"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit precision"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ngohongthai/exam-sixth_grade-instruct-dataset",
        help="Dataset name or path"
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Training split name"
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        help="Evaluation split name"
    )
    
    # Training settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gemma3n-finetune",
        help="Output directory"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline",
        help="Experiment name"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per device training batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    
    # LoRA settings
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=8,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout"
    )
    
    # System settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (auto-detected if not specified)"
    )
    
    # Logging and tracking
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    parser.add_argument(
        "--disable-comet",
        action="store_true",
        help="Disable Comet ML tracking"
    )
    
    # Model saving
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Save trained model"
    )
    parser.add_argument(
        "--save-formats",
        nargs="+",
        default=["lora"],
        choices=["lora", "merged_fp16", "merged_bf16", "gguf_q8", "gguf_f16"],
        help="Model save formats"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-username",
        type=str,
        help="HuggingFace Hub username"
    )
    
    # Testing
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Test model after training"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test with minimal steps"
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[TrainingConfig, CometConfig]:
    """Load configuration from file or command line arguments."""
    
    # Load from config file if provided
    if args.config:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            training_config = TrainingConfig.from_yaml(args.config)
        elif args.config.endswith('.json'):
            training_config = TrainingConfig.from_json(args.config)
        else:
            raise ValueError(f"Unsupported config file format: {args.config}")
    else:
        # Create config from command line arguments
        training_config = TrainingConfig()
    
    # Override with command line arguments
    override_dict = {}
    
    # Model settings
    if args.model_name != "unsloth/gemma-3n-E4B-it":
        override_dict["model_name"] = args.model_name
    if args.max_seq_length != 2048:
        override_dict["max_seq_length"] = args.max_seq_length
    if not args.load_in_4bit:
        override_dict["load_in_4bit"] = args.load_in_4bit
    
    # Dataset settings
    if args.dataset_name != "ngohongthai/exam-sixth_grade-instruct-dataset":
        override_dict["dataset_name"] = args.dataset_name
    if args.train_split != "train":
        override_dict["train_split"] = args.train_split
    
    # Training settings
    if args.output_dir != "outputs/gemma3n-finetune":
        override_dict["output_dir"] = args.output_dir
    if args.experiment_name != "baseline":
        override_dict["experiment_name"] = args.experiment_name
    if args.max_steps != 100:
        override_dict["max_steps"] = args.max_steps
    if args.batch_size != 1:
        override_dict["per_device_train_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps != 8:
        override_dict["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate != 2e-4:
        override_dict["learning_rate"] = args.learning_rate
    
    # LoRA settings
    if args.lora_r != 8:
        override_dict["lora_r"] = args.lora_r
    if args.lora_alpha != 8:
        override_dict["lora_alpha"] = args.lora_alpha
    if args.lora_dropout != 0.0:
        override_dict["lora_dropout"] = args.lora_dropout
    
    # System settings
    if args.seed != 42:
        override_dict["seed"] = args.seed
    
    # Apply overrides
    for key, value in override_dict.items():
        setattr(training_config, key, value)
    
    # Quick test configuration
    if args.quick_test:
        training_config.max_steps = 10
        training_config.logging_steps = 1
        training_config.save_steps = 5
        training_config.experiment_name = "quick_test"
    
    # Disable Comet if requested
    if args.disable_comet:
        training_config.report_to = "none"
    
    # Create Comet config
    comet_config = CometConfig()
    if args.disable_comet:
        comet_config.api_key = None
    
    return training_config, comet_config


def setup_environment(args: argparse.Namespace, training_config: TrainingConfig) -> None:
    """Setup environment and logging."""
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    logger = get_logger()
    logger.info("🚀 Starting Gemma3N training pipeline")
    
    # Set random seed
    TrainingUtils.set_seed(training_config.seed)
    
    # Print device information
    DeviceUtils.print_device_info()
    
    # Setup output directory
    TrainingUtils.setup_output_directory(training_config.get_output_dir())
    
    # Save configuration
    config_path = os.path.join(training_config.get_output_dir(), "training_config.json")
    training_config.save_json(config_path)
    logger.info(f"💾 Configuration saved to {config_path}")


def main() -> None:
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    training_config, comet_config = load_config(args)
    
    # Validate configuration
    training_config.validate()
    
    # Setup environment
    setup_environment(args, training_config)
    
    logger = get_logger()
    start_time = time.time()
    
    try:
        # Initialize experiment tracking
        comet_tracker = None
        if training_config.report_to == "comet_ml":
            comet_tracker = CometTracker(comet_config)
            comet_tracker.setup_experiment(
                training_config=training_config.to_dict(),
                model_config=training_config.to_lora_config_kwargs()
            )
        
        # Load model
        logger.info("📂 Loading model and tokenizer...")
        model_loader = ModelLoader(training_config)
        model, tokenizer = model_loader.load_complete_model(apply_lora=True)
        model_loader.print_model_info(model)
        
        # Load and prepare dataset
        logger.info("📊 Loading and preparing dataset...")
        dataset_processor = DatasetProcessor(tokenizer)
        
        datasets = dataset_processor.prepare_datasets(
            dataset_name=training_config.dataset_name,
            train_split=training_config.train_split,
            eval_split=args.eval_split if args.eval_split else None
        )
        
        train_dataset = datasets["train"]
        eval_dataset = datasets.get("eval")
        
        # Preview dataset
        dataset_processor.preview_dataset(train_dataset, num_samples=2)
        
        # Create trainer
        logger.info("🏋️ Creating trainer...")
        trainer_factory = TrainerFactory(training_config)
        
        pipeline = trainer_factory.create_training_pipeline(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer = pipeline["trainer"]
        trainer_factory.print_training_info(trainer, train_dataset, eval_dataset)
        
        # Start training
        logger.info("🚀 Starting training...")
        trainer_stats = trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Log training summary to Comet
        if comet_tracker and comet_tracker.is_active():
            comet_tracker.log_training_summary(trainer_stats.__dict__)
        
        # Save model
        if args.save_model:
            logger.info("💾 Saving model...")
            model_saver = ModelSaver(model, tokenizer)
            
            # Prepare save formats
            formats = {}
            for fmt in args.save_formats:
                if fmt == "lora":
                    formats["lora"] = {}
                elif fmt.startswith("merged_"):
                    precision = fmt.split("_")[1]
                    formats[fmt] = {"precision": precision}
                elif fmt.startswith("gguf_"):
                    quantization = fmt.split("_")[1]
                    formats[fmt] = {"quantization": quantization}
            
            # Save in requested formats
            save_results = model_saver.save_all_formats(
                base_save_path=training_config.get_output_dir(),
                model_name=f"gemma3n-{training_config.experiment_name}",
                formats=formats,
                push_to_hub=args.push_to_hub,
                hub_username=args.hub_username,
                token=os.getenv("HF_TOKEN")
            )
            
            logger.info("💾 Model saving results:")
            for fmt, path in save_results.items():
                logger.info(f"   {fmt}: {path}")
            
            # Log model to Comet
            if comet_tracker and comet_tracker.is_active():
                for fmt, path in save_results.items():
                    if not path.startswith("Error"):
                        comet_tracker.log_model(path, f"model_{fmt}")
        
        # Test model
        if args.test_model:
            logger.info("🧪 Testing model...")
            inference_engine = InferenceEngine(model, tokenizer)
            test_results = inference_engine.test_model()
            
            logger.info("🧪 Test Results:")
            for i, result in enumerate(test_results, 1):
                logger.info(f"   Test {i}:")
                logger.info(f"      Q: {result['question']}")
                logger.info(f"      A: {result['answer']}")
        
        # End experiment tracking
        if comet_tracker and comet_tracker.is_active():
            comet_tracker.end_experiment()
        
        logger.info(f"🎉 Training pipeline completed successfully in {training_time:.2f} seconds!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if comet_tracker and comet_tracker.is_active():
            comet_tracker.end_experiment()
        raise
    finally:
        # Cleanup
        DeviceUtils.clear_cuda_cache()


if __name__ == "__main__":
    main()