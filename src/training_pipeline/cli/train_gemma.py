"""
MathPal Training Pipeline - Enhanced Config Architecture
========================================================

Production-ready training CLI using the new ConfigManager system for unified configuration management.
Provides type-safe config access, proper dependency injection, and comprehensive validation.

Usage:
    python -m training_pipeline.cli.train_gemma --config configs/quick_test.yaml
    python -m training_pipeline.cli.train_gemma --config configs/production.yaml --experiment-name my-experiment
    python -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml --quick-test

Key Features:
    ✅ Unified config management across all formats
    ✅ Type-safe config section access  
    ✅ Better error messages and debugging
    ✅ Proper dependency injection for managers
    ✅ Environment variable and CLI override support
    ✅ Quick test and production modes
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Fix tokenizer verbose logging before importing transformers
import logging
import warnings

# Environment variables to suppress verbose output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import and configure transformers logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

# Configure all related loggers to ERROR level
loggers_to_suppress = [
    "transformers", "transformers.tokenization_utils_base", 
    "transformers.tokenization_utils", "transformers.models",
    "transformers.generation", "tokenizers", "unsloth",
    "datasets", "accelerate", "bitsandbytes", "peft"
]

for logger_name in loggers_to_suppress:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Set root logging to WARNING to reduce spam
logging.getLogger().setLevel(logging.WARNING)

from training_pipeline.config.config_manager import create_config_manager, ConfigManager
from training_pipeline.core.training_manager import TrainingManager, TrainingResults
from training_pipeline.core.exceptions import (
    ConfigurationError, ValidationError, TrainingError, ModelError, DatasetError
)
from training_pipeline.utils import setup_logging, get_logger, DeviceUtils

# Global logger variable
logger = None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MathPal Training Pipeline - Enhanced Config Management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
    # Quick development test
    python -m training_pipeline.cli.train_gemma --config configs/quick_test.yaml
    
    # Production training
    python -m training_pipeline.cli.train_gemma --config configs/production.yaml
    
    # Custom training with overrides
    python -m training_pipeline.cli.train_gemma --config configs/production.yaml \\
        --experiment-name my-custom-experiment --max-steps 1000 --learning-rate 1e-4
    
    # Quick test with any config
    python -m training_pipeline.cli.train_gemma --config configs/production.yaml --quick-test
        """
    )
    
    # Core arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    
    # Optional overrides
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override experiment name from config"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override maximum training steps from config"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name from config"
    )
    
    # Special modes
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal steps (20 steps, small batch)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and estimate resources without training"
    )
    
    # System settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Disable Comet ML tracking even if configured"
    )
    
    parser.add_argument(
        "--no-env",
        action="store_true", 
        help="Disable environment variable overrides"
    )
    
    return parser.parse_args()


def load_and_validate_config(args: argparse.Namespace) -> ConfigManager:
    """
    Load configuration using new ConfigManager system.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Validated ConfigManager instance
        
    Raises:
        ConfigurationError: If config loading or validation fails
    """
    try:
        logger.info(f"📄 Loading configuration from: {args.config}")
        
        # Prepare CLI overrides
        cli_overrides = {}
        if args.experiment_name:
            cli_overrides['experiment_name'] = args.experiment_name
        if args.output_dir:
            cli_overrides['output_dir'] = args.output_dir
        if args.max_steps:
            cli_overrides['max_steps'] = args.max_steps
        if args.learning_rate:
            cli_overrides['learning_rate'] = args.learning_rate
        if args.model_name:
            cli_overrides['model_name'] = args.model_name
        
        # Apply quick test overrides
        if args.quick_test:
            logger.info("⚡ Applying quick test profile...")
            cli_overrides.update({
                'max_steps': 20,
                'per_device_train_batch_size': 1,
                'gradient_accumulation_steps': 4,
                'max_seq_length': 1024,
                'experiment_name': 'quick-test',
                'save_steps': 10,
                'logging_steps': 2,
                'eval_steps': 10,
            })
        
        # Disable Comet if requested
        if args.no_comet:
            cli_overrides['report_to'] = 'none'
        
        # Create ConfigManager with all overrides
        config_manager = create_config_manager(
            config_path=args.config,
            apply_env=not args.no_env,
            cli_overrides=cli_overrides if cli_overrides else None
        )
        
        logger.info("✅ Configuration loaded and validated successfully")
        logger.info(config_manager.summary())
        
        return config_manager
        
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {args.config}")
    except ValidationError as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}")


def setup_environment(args: argparse.Namespace, config_manager: ConfigManager) -> None:
    """
    Setup logging and environment.
    
    Args:
        args: Command line arguments  
        config_manager: Configuration manager
    """
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level)
    
    # Update logger reference
    global logger
    logger = get_logger("mathpal_training")
    
    logger.info("🚀 MathPal Training Pipeline - Starting...")
    logger.info("=" * 60)
    logger.info("📋 Configuration Summary:")
    logger.info(f"   🤖 Model: {config_manager.model.name}")
    logger.info(f"   📚 Dataset: {config_manager.dataset.name}")
    logger.info(f"   🎯 Max steps: {config_manager.training.max_steps:,}")
    logger.info(f"   📦 Batch size: {config_manager.training.per_device_train_batch_size}")
    logger.info(f"   📈 Learning rate: {config_manager.training.learning_rate:.2e}")
    logger.info(f"   🎲 LoRA rank: {config_manager.lora.r}")
    logger.info(f"   📁 Output: {config_manager.output.get_output_dir()}")
    logger.info(f"   📊 Effective batch size: {config_manager.get_effective_batch_size()}")
    logger.info(f"   🔬 Comet tracking: {config_manager.comet.enabled}")
    logger.info("=" * 60)


def run_dry_run(config_manager: ConfigManager) -> None:
    """
    Run dry run mode to validate config and estimate resources.
    
    Args:
        config_manager: Configuration manager
    """
    logger.info("🔍 Running dry run mode...")
    
    try:
        # Validate all configuration sections
        config_manager.validate_all()
        logger.info("✅ All configuration sections validated")
        
        # Print detailed config summary
        logger.info("📊 Configuration Details:")
        
        logger.info("🤖 Model Configuration:")
        model_config = config_manager.model
        logger.info(f"   Name: {model_config.name}")
        logger.info(f"   Max length: {model_config.max_seq_length}")
        logger.info(f"   4-bit quantization: {model_config.load_in_4bit}")
        logger.info(f"   Full fine-tuning: {model_config.full_finetuning}")
        
        logger.info("📚 Dataset Configuration:")
        dataset_config = config_manager.dataset
        logger.info(f"   Name: {dataset_config.name}")
        logger.info(f"   Train split: {dataset_config.train_split}")
        logger.info(f"   Text field: {dataset_config.text_field}")
        
        logger.info("🎯 Training Configuration:")
        training_config = config_manager.training
        logger.info(f"   Max steps: {training_config.max_steps}")
        logger.info(f"   Learning rate: {training_config.learning_rate:.2e}")
        logger.info(f"   Batch size: {training_config.per_device_train_batch_size}")
        logger.info(f"   Effective batch size: {config_manager.get_effective_batch_size()}")
        
        logger.info("🔧 LoRA Configuration:")
        lora_config = config_manager.lora
        logger.info(f"   Rank: {lora_config.r}")
        logger.info(f"   Alpha: {lora_config.alpha}")
        logger.info(f"   Dropout: {lora_config.dropout}")
        logger.info(f"   Target modules: {lora_config.target_modules}")
        
        logger.info("📁 Output Configuration:")
        output_config = config_manager.output
        logger.info(f"   Directory: {output_config.get_output_dir()}")
        logger.info(f"   Save formats: {output_config.save_formats}")
        logger.info(f"   Save steps: {output_config.save_steps}")
        
        # Check system requirements
        logger.info("🖥️ System Information:")
        DeviceUtils.print_device_info()
        
        logger.info("✅ Dry run completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Dry run failed: {e}")
        raise


def run_training(config_manager: ConfigManager) -> None:
    """
    Run the complete training pipeline with new config system.
    
    Args:
        config_manager: Configuration manager
    """
    try:
        # Use ConfigManager directly with new TrainingManager
        from training_pipeline.core.training_manager import TrainingManager
        manager = TrainingManager(config_manager)
        
        # Run training pipeline
        results = manager.run_training()
        
        # Print final results
        logger.info("🎉 Training completed successfully!")
        logger.info("📊 Final Results:")
        logger.info(f"   📉 Final loss: {results.final_loss:.4f}")
        logger.info(f"   ⏱️ Training time: {results.training_time:.2f} seconds")
        logger.info(f"   📈 Total steps: {results.total_steps:,}")
        logger.info(f"   💾 Models saved: {len(results.model_paths)}")
        
        # Show saved model paths
        for format_name, path in results.model_paths.items():
            if not path.startswith("Error"):
                logger.info(f"      📦 {format_name}: {path}")
        
        logger.info("=" * 60)
        logger.info("🎊 MathPal Training Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def main() -> None:
    """Main entry point for training pipeline."""
    exit_code = 0
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup basic logging first (before config loading)
        setup_logging(log_level="DEBUG" if args.debug else "INFO")
        global logger
        logger = get_logger("mathpal_training")
        
        # Load and validate configuration using new ConfigManager
        config_manager = load_and_validate_config(args)
        
        # Setup full environment and logging
        setup_environment(args, config_manager)
        
        # Run appropriate mode
        if args.dry_run:
            run_dry_run(config_manager)
        else:
            run_training(config_manager)
            
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}", file=sys.stderr)
        print("💡 Please check your configuration file and try again.", file=sys.stderr)
        exit_code = 1
        
    except ValidationError as e:
        print(f"❌ Validation Error: {e}", file=sys.stderr)
        print("💡 Please fix the validation errors and try again.", file=sys.stderr)
        exit_code = 1
        
    except ModelError as e:
        print(f"❌ Model Error: {e}", file=sys.stderr)
        print("💡 Check model name and ensure required libraries are installed.", file=sys.stderr)
        exit_code = 2
        
    except DatasetError as e:
        print(f"❌ Dataset Error: {e}", file=sys.stderr)
        print("💡 Check dataset name and internet connection.", file=sys.stderr)
        exit_code = 3
        
    except TrainingError as e:
        print(f"❌ Training Error: {e}", file=sys.stderr)
        print("💡 Check logs for detailed error information.", file=sys.stderr)
        exit_code = 4
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user", file=sys.stderr)
        exit_code = 130
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}", file=sys.stderr)
        print("💡 Please report this issue with full error details.", file=sys.stderr)
        exit_code = 5
        
    finally:
        # Cleanup
        try:
            DeviceUtils.clear_cuda_cache()
        except:
            pass
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
