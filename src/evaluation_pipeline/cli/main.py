#!/usr/bin/env python3
"""
Main CLI entry point for the evaluation pipeline.
"""

import click
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation_pipeline.utils import setup_logging
from src.evaluation_pipeline.config import ConfigManager
from src.evaluation_pipeline.utils.logger import (
    setup_logger as setup_rich_logger, 
    print_evaluation_header, 
    print_evaluation_results,
    print_model_info,
    print_dataset_info,
    print_step_header,
    print_error_summary,
    print_success_message
)


@click.group()
@click.option(
    '--config', 
    '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
    default='INFO',
    help='Logging level'
)
@click.option(
    '--log-file',
    type=click.Path(path_type=Path),
    help='Log file path'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.version_option(version='1.0.0', prog_name='MathPal Evaluation Pipeline')
@click.pass_context
def cli(ctx, config, log_level, log_file, verbose):
    """
    MathPal Evaluation Pipeline - Vietnamese Math AI Model Evaluation
    
    A comprehensive evaluation pipeline for Vietnamese math education AI models.
    Supports Gemma 3N, Opik metrics, OpenRouter LLM-as-a-judge, and custom Vietnamese math metrics.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    if verbose:
        log_level = 'DEBUG'
    
    logger = setup_logging(
        level=log_level,
        log_file=log_file
    )
    
    # Load configuration
    try:
        if config:
            config_manager = ConfigManager.from_yaml(config)
        else:
            config_manager = ConfigManager.create_default("cli")
        
        ctx.obj['config'] = config_manager
        ctx.obj['logger'] = logger
        
        logger.info("CLI initialized successfully")
        
    except Exception as e:
        click.echo(f"Error initializing CLI: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--model-path',
    '-m',
    type=str,
    required=True,
    help='Path to the model to evaluate (local path or Hugging Face model name)'
)
@click.option(
    '--dataset',
    '-d',
    type=str,
    help='Path to evaluation dataset (JSON), Hugging Face dataset ID (e.g., "username/dataset_name"), or predefined dataset name (e.g., "ngohongthai")'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    default='evaluation_results.json',
    help='Output file path for results'
)
@click.option(
    '--mode',
    type=click.Choice(['quick', 'comprehensive'], case_sensitive=False),
    default='comprehensive',
    help='Evaluation mode'
)
@click.option(
    '--batch-size',
    type=int,
    default=8,
    help='Batch size for model inference'
)
@click.option(
    '--max-samples',
    type=int,
    help='Maximum number of samples to evaluate'
)
@click.option(
    '--save-predictions',
    is_flag=True,
    help='Save model predictions to file'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Run without actual model evaluation (for testing)'
)
@click.pass_context
def evaluate(ctx, model_path, dataset, output, mode, batch_size, max_samples, save_predictions, dry_run):
    """
    Evaluate a Vietnamese math AI model.
    
    This command evaluates a model using comprehensive metrics including:
    - Opik metrics (hallucination, context precision, etc.)
    - Vietnamese math-specific metrics
    - LLM-as-a-judge evaluation via OpenRouter
    - Custom educational metrics
    """
    from src.evaluation_pipeline.managers import EvaluationManager
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    # Print evaluation header
    print_evaluation_header()
    
    eval_manager = None
    try:
        # Step 1: Create evaluation manager
        print_step_header("Initializing Evaluation Manager", 1, 5)
        eval_manager = EvaluationManager(config)
        print_success_message("Evaluation manager initialized successfully")
        
        # Step 2: Load dataset
        print_step_header("Loading Dataset", 2, 5)
        if dataset:
            # Check if it's a predefined dataset name
            predefined_datasets = config.config.dataset.predefined
            if dataset in predefined_datasets:
                samples = eval_manager.dataset_manager.load_predefined_dataset(dataset)
            else:
                samples = eval_manager.dataset_manager.load_dataset(dataset)
        else:
            samples = eval_manager.dataset_manager.get_default_dataset()
        
        # Print dataset info
        print_dataset_info(
            dataset_name=dataset or "Default Dataset",
            sample_count=len(samples),
            source=config.config.dataset.source
        )
        
        # Limit samples if specified
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]
            logger.info(f"Limited evaluation to {max_samples} samples")
        
        # Update config for this run
        config.config.model.batch_size = batch_size
        config.config.evaluation.mode = mode
        config.config.evaluation.save_predictions = save_predictions
        
        if dry_run:
            logger.info("DRY RUN MODE - No actual evaluation will be performed")
            print_warning_message("Dry run completed successfully")
            return
        
        # Step 3: Print model info
        print_step_header("Model Information", 3, 5)
        print_model_info(
            model_name=model_path.split("/")[-1] if "/" in model_path else model_path,
            model_path=model_path,
            device=config.config.hardware.device
        )
        
        # Step 4: Run evaluation
        print_step_header("Running Evaluation", 4, 5)
        logger.info(f"Starting evaluation of {len(samples)} samples")
        results = eval_manager.evaluate_model(
            model_path=model_path,
            samples=samples
        )
        
        # Step 5: Save and display results
        print_step_header("Saving Results", 5, 5)
        eval_manager.save_results(results)
        
        # Display results with beautiful formatting
        print_evaluation_results(
            results=results.metrics,
            model_name=model_path.split("/")[-1] if "/" in model_path else model_path
        )
        
        # Display summary
        print_success_message(f"Evaluation completed successfully! Results saved to: {output}")
        
        # Show top metrics
        click.echo(f"\n🏆 Top Metrics:")
        sorted_metrics = sorted(
            results.metrics.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        for metric, score in sorted_metrics:
            click.echo(f"  {metric}: {score:.3f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print_error_summary(str(e), "Evaluation process encountered an error")
        # Cleanup resources
        if eval_manager is not None:
            try:
                eval_manager.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure cleanup happens even on success
        if eval_manager is not None:
            try:
                eval_manager.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")


@cli.command()
@click.option(
    '--dataset-name',
    type=str,
    help='Name of predefined dataset to show info for'
)
@click.pass_context
def dataset_info(ctx, dataset_name):
    """
    Show information about available datasets.
    
    Shows information about:
    - Default dataset configuration
    - Predefined datasets
    - Dataset field mappings
    """
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    try:
        from src.evaluation_pipeline.managers import EvaluationManager
        eval_manager = EvaluationManager(config)
        
        click.echo("📚 Dataset Information")
        click.echo("="*50)
        
        # Show default dataset info
        default_info = eval_manager.dataset_manager.get_dataset_info()
        click.echo(f"Default Dataset:")
        click.echo(f"  Source: {default_info['source']}")
        click.echo(f"  ID: {default_info['id']}")
        click.echo(f"  Split: {default_info['split']}")
        
        if 'info' in default_info:
            info = default_info['info']
            click.echo(f"  Name: {info['name']}")
            click.echo(f"  Description: {info['description']}")
            click.echo(f"  Total Samples: {info['total_samples']}")
            click.echo(f"  Grade Levels: {', '.join(info['grade_levels'])}")
            click.echo(f"  Subjects: {', '.join(info['subjects'])}")
            click.echo(f"  Language: {info['language']}")
            click.echo(f"  Source: {info['source']}")
        
        click.echo()
        
        # Show predefined datasets
        predefined_datasets = config.config.dataset.predefined
        click.echo("Predefined Datasets:")
        for name, info in predefined_datasets.items():
            click.echo(f"  {name}:")
            click.echo(f"    ID: {info['id']}")
            click.echo(f"    Split: {info['split']}")
            click.echo(f"    Description: {info['description']}")
            click.echo(f"    Samples: {info['samples']}")
            click.echo(f"    Grade Level: {info['grade_level']}")
            click.echo(f"    Subject: {info['subject']}")
            click.echo()
        
        # Show field mapping
        field_mapping = config.config.dataset.field_mapping
        click.echo("Field Mapping (Hugging Face → Internal):")
        for internal_field, huggingface_fields in field_mapping.items():
            click.echo(f"  {internal_field}: {', '.join(huggingface_fields)}")
        
        # Show specific dataset info if requested
        if dataset_name:
            click.echo()
            click.echo(f"📋 Detailed Info for '{dataset_name}':")
            specific_info = eval_manager.dataset_manager.get_dataset_info(dataset_name)
            if 'error' in specific_info:
                click.echo(f"  ❌ {specific_info['error']}")
            else:
                for key, value in specific_info.items():
                    click.echo(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error showing dataset info: {e}")
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose test output'
)
@click.pass_context
def test(ctx, verbose):
    """
    Run tests for the evaluation pipeline.
    
    Runs comprehensive tests for:
    - Configuration management
    - Core imports and dependencies
    - CLI interface functionality
    """
    import subprocess
    import sys
    
    click.echo("🧪 Running evaluation pipeline tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "src/evaluation_pipeline/test_evaluation.py"
        ], cwd=project_root, capture_output=not verbose)
        
        if result.returncode == 0:
            click.echo("✅ All tests passed")
        else:
            click.echo("❌ Some tests failed")
            if verbose:
                click.echo(result.stdout.decode())
                click.echo(result.stderr.decode())
                
    except Exception as e:
        click.echo(f"❌ Error running tests: {e}")


@cli.command()
@click.option(
    '--show-defaults',
    is_flag=True,
    help='Show default configuration values'
)
@click.option(
    '--validate',
    is_flag=True,
    help='Validate current configuration'
)
@click.option(
    '--create-default',
    type=click.Path(path_type=Path),
    help='Create default configuration file'
)
@click.pass_context
def config(ctx, show_defaults, validate, create_default):
    """
    Manage evaluation pipeline configuration.
    """
    config = ctx.obj['config']
    
    if show_defaults:
        click.echo("📋 Default Configuration:")
        click.echo(config.config.model_dump_json(indent=2))
    
    elif validate:
        try:
            # Validate configuration
            config.validate()
            click.echo("✅ Configuration is valid")
        except Exception as e:
            click.echo(f"❌ Configuration validation failed: {e}")
    
    elif create_default:
        try:
            config.save_config(create_default)
            click.echo(f"✅ Default configuration created at: {create_default}")
        except Exception as e:
            click.echo(f"❌ Failed to create configuration: {e}")
    
    else:
        click.echo("📋 Current Configuration:")
        click.echo(config.config.model_dump_json(indent=2))


@cli.command()
@click.pass_context
def info(ctx):
    """
    Display information about the evaluation pipeline.
    """
    click.echo("🤖 MathPal Evaluation Pipeline")
    click.echo("=" * 50)
    click.echo("A comprehensive evaluation pipeline for Vietnamese math education AI models.")
    click.echo()
    
    click.echo("📊 Supported Metrics:")
    click.echo("  • Opik Metrics (hallucination, context precision, etc.)")
    click.echo("  • Vietnamese Math Metrics (accuracy, language quality, etc.)")
    click.echo("  • LLM-as-a-Judge (via OpenRouter)")
    click.echo("  • Custom Educational Metrics")
    click.echo()
    
    click.echo("🚀 Supported Models:")
    click.echo("  • Gemma 3N with MatFormer optimization")
    click.echo("  • Custom model integration")
    click.echo()
    
    click.echo("🔧 Features:")
    click.echo("  • Batch processing with memory optimization")
    click.echo("  • Streaming generation support")
    click.echo("  • Rate limiting and cost tracking")
    click.echo("  • Comprehensive logging and monitoring")
    click.echo()
    
    click.echo("📁 Project Structure:")
    click.echo("  • Modular architecture")
    click.echo("  • Configuration management")
    click.echo("  • CLI interface")
    click.echo("  • Production-ready evaluation")
    click.echo()
    
    click.echo("🎯 Use Cases:")
    click.echo("  • Vietnamese math education model evaluation")
    click.echo("  • Grade 5-6 transition assessment")
    click.echo("  • Educational AI quality assurance")
    click.echo("  • Research and development validation")
    click.echo()
    
    click.echo("💡 Usage Examples:")
    click.echo("  # Evaluate model with default config")
    click.echo("  python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model")
    click.echo()
    click.echo("  # Evaluate with custom config")
    click.echo("  python -m src.evaluation_pipeline.cli.main -c config.yaml evaluate -m /path/to/model")
    click.echo()
    click.echo("  # Quick evaluation with limited samples")
    click.echo("  python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model --mode quick --max-samples 10")


if __name__ == '__main__':
    cli()
