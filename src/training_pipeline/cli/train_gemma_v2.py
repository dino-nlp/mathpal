"""
MathPal Training Pipeline v2 - New Architecture
===============================================

Simplified, modular training CLI for Vietnamese math tutoring models.
Uses config-driven approach with comprehensive YAML configuration.

Usage:
    python -m training_pipeline.cli.train_gemma_v2 --config configs/complete_training_config.yaml
    python -m training_pipeline.cli.train_gemma_v2 --config configs/development.yaml --quick-test
    python -m training_pipeline.cli.train_gemma_v2 --config configs/production.yaml --experiment-name my-experiment

Features:
    ✅ Config-driven approach (YAML files)
    ✅ Only 5-7 CLI arguments vs 50+ in old version  
    ✅ Modular architecture with factories and managers
    ✅ Comprehensive error handling and validation
    ✅ Built-in experiment tracking (Comet ML)
    ✅ Multiple model save formats
    ✅ Vietnamese math-specific evaluation
    ✅ Memory estimation and optimization
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training_pipeline.core.enhanced_config import ConfigLoader, ComprehensiveTrainingConfig
from training_pipeline.core.training_manager import TrainingManager, TrainingResults
from training_pipeline.core.exceptions import (
    ConfigurationError, ValidationError, TrainingError, ModelError, DatasetError
)
from training_pipeline.utils import setup_logging, get_logger, DeviceUtils


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    New architecture uses minimal CLI arguments with config-driven approach.
    """
    parser = argparse.ArgumentParser(
        description="MathPal Training Pipeline v2 - Train Gemma3N for Vietnamese Math Tutoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
    # Development training with quick test
    python -m training_pipeline.cli.train_gemma_v2 --config configs/development.yaml --quick-test
    
    # Production training with custom experiment name
    python -m training_pipeline.cli.train_gemma_v2 --config configs/production.yaml --experiment-name production-v2
    
    # Custom training with overrides
    python -m training_pipeline.cli.train_gemma_v2 --config configs/complete_training_config.yaml --max-steps 500 --output-dir outputs/custom
        """
    )
    
    # Core arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    
    # Optional overrides (only the most important ones)
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
    
    # Special modes
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal steps (overrides config)"
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
    
    return parser.parse_args()


def load_and_validate_config(args: argparse.Namespace) -> ComprehensiveTrainingConfig:
    """
    Load configuration from file and apply CLI overrides.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Validated training configuration
        
    Raises:
        ConfigurationError: If config loading or validation fails
    """
    try:
        # Load base configuration from YAML
        logger.info(f"📄 Loading configuration from: {args.config}")
        config = ConfigLoader.load_from_yaml(args.config)
        
        # Apply quick test profile if requested
        if args.quick_test:
            logger.info("⚡ Applying quick test profile...")
            config = ConfigLoader.apply_quick_test_profile(config)
        
        # Apply CLI overrides
        config = ConfigLoader.apply_cli_overrides(
            config,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            max_steps=args.max_steps
        )
        
        # Disable Comet if requested
        if args.no_comet:
            config.comet.enabled = False
            config.logging.report_to = "none"
        
        # Validate configuration
        logger.info("🔍 Validating configuration...")
        config.validate()
        
        logger.info("✅ Configuration loaded and validated successfully")
        return config
        
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {args.config}")
    except ValidationError as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}")


def setup_environment(args: argparse.Namespace, config: ComprehensiveTrainingConfig) -> None:
    """
    Setup logging and environment.
    
    Args:
        args: Command line arguments  
        config: Training configuration
    """
    # Determine log level
    log_level = "DEBUG" if args.debug else config.logging.level
    
    # Setup logging
    setup_logging(
        log_level=log_level,
        log_file=config.logging.log_file
    )
    
    # Create logger after setup
    global logger
    logger = get_logger()
    
    logger.info("🚀 MathPal Training Pipeline v2 - Starting...")
    logger.info("=" * 60)
    logger.info("📋 Configuration Summary:")
    logger.info(f"   🤖 Model: {config.model.name}")
    logger.info(f"   📚 Dataset: {config.dataset.name}")
    logger.info(f"   🎯 Max steps: {config.training.max_steps:,}")
    logger.info(f"   📦 Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"   📈 Learning rate: {config.training.learning_rate:.2e}")
    logger.info(f"   🎲 LoRA rank: {config.lora.r}")
    logger.info(f"   📁 Output: {config.get_output_dir()}")
    logger.info("=" * 60)


def run_dry_run(config: ComprehensiveTrainingConfig) -> None:
    """
    Run dry run mode to validate config and estimate resources.
    
    Args:
        config: Training configuration
    """
    logger.info("🔍 Running dry run mode...")
    
    try:
        # Create training manager
        manager = TrainingManager(config)
        
        # Validate configuration
        manager.validate_config()
        
        # Estimate training costs
        cost_estimate = manager.estimate_training_cost()
        
        logger.info("💰 Training Cost Estimation:")
        logger.info(f"   💾 Estimated memory: {cost_estimate.get('estimated_memory_gb', 'Unknown'):.1f} GB")
        logger.info(f"   💾 Available memory: {cost_estimate.get('available_memory_gb', 'Unknown'):.1f} GB")
        logger.info(f"   📊 Memory utilization: {cost_estimate.get('memory_utilization', 0)*100:.1f}%")
        logger.info(f"   ⏱️ Estimated time: {cost_estimate.get('estimated_time_hours', 'Unknown'):.1f} hours")
        logger.info(f"   ✅ Feasible: {'Yes' if cost_estimate.get('feasible', True) else 'No'}")
        
        # Check system requirements
        logger.info("🖥️ System Information:")
        DeviceUtils.print_device_info()
        
        # Check dataset availability
        from training_pipeline.factories import DatasetFactory
        dataset_info = DatasetFactory.get_dataset_info(config)
        logger.info(f"📊 Dataset: {dataset_info.get('name', 'Unknown')}")
        logger.info(f"   ✅ Supported: {'Yes' if dataset_info.get('supported', False) else 'No'}")
        
        if not cost_estimate.get('feasible', True):
            logger.warning("⚠️ Training may not be feasible with current configuration!")
            logger.warning("Consider:")
            logger.warning("   - Reducing batch size")
            logger.warning("   - Enabling 4-bit quantization")
            logger.warning("   - Using gradient checkpointing")
            logger.warning("   - Reducing sequence length")
        
        logger.info("✅ Dry run completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Dry run failed: {e}")
        raise


def run_training(config: ComprehensiveTrainingConfig) -> None:
    """
    Run the complete training pipeline.
    
    Args:
        config: Training configuration
    """
    try:
        # Create training manager
        manager = TrainingManager(config)
        
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
        
        # Show evaluation results if available
        if results.evaluation_results:
            logger.info(f"   🧪 Evaluation completed: {len(results.evaluation_results)} test suites")
            
            # Show inference test results
            inference_tests = results.evaluation_results.get("inference_tests", [])
            successful_tests = sum(1 for test in inference_tests if test.get("status") == "success")
            logger.info(f"      ✅ Inference tests passed: {successful_tests}/{len(inference_tests)}")
            
            # Show Vietnamese math test results
            vn_math_tests = results.evaluation_results.get("vietnamese_math_tests", [])
            if vn_math_tests:
                avg_quality = sum(test.get("quality_score", 0) for test in vn_math_tests) / len(vn_math_tests)
                logger.info(f"      🇻🇳 Vietnamese math quality: {avg_quality:.2f}/1.0")
        
        logger.info("=" * 60)
        logger.info("🎊 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def main() -> None:
    """Main entry point for training pipeline."""
    exit_code = 0
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load and validate configuration  
        config = load_and_validate_config(args)
        
        # Setup environment and logging
        setup_environment(args, config)
        
        # Run appropriate mode
        if args.dry_run:
            run_dry_run(config)
        else:
            run_training(config)
            
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
