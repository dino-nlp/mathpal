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
    print_success_message,
    print_warning_message
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
    
    # Load configuration first to get logging format
    try:
        if config:
            config_manager = ConfigManager.from_yaml(config)
        else:
            raise ValueError("No configuration file provided")
        
        # Get logging format from config
        logging_config = config_manager.get_logging_config()
        log_format = logging_config.format
        
        logger = setup_logging(
            level=log_level,
            log_file=log_file,
            format=log_format
        )
        
        ctx.obj['config'] = config_manager
        ctx.obj['logger'] = logger
        
        logger.info("CLI initialized successfully")
        
    except Exception as e:
        # Fallback to default logging if config loading fails
        logger = setup_logging(
            level=log_level,
            log_file=log_file,
            format="text"  # Default to text format
        )
        click.echo(f"Error initializing CLI: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--model-path',
    '-m',
    type=str,
    help='Path to the model to evaluate (local path or Hugging Face model name). If not provided, will use model from config file.'
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
    '--batch-size',
    type=int,
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
@click.pass_context
def evaluate(ctx, model_path, dataset, output, batch_size, max_samples, save_predictions):
    """
    Evaluate a Vietnamese math AI model.
    
    This command evaluates a model using comprehensive metrics including:
    - Opik metrics (hallucination, context precision, etc.)
    - Vietnamese math-specific metrics
    - LLM-as-a-judge evaluation via OpenRouter
    - Custom educational metrics
    
    Configuration Priority:
    - Command line arguments override config file settings
    - If command line argument is not provided, uses value from config file
    - If neither is available, uses sensible defaults
    """
    from src.evaluation_pipeline.managers import EvaluationManager
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    # Print evaluation header
    print_evaluation_header()
    eval_manager = None
    try:
        # Step 1: Create evaluation manager
        print_step_header("Initializing Evaluation Manager", 1, 6)
        eval_manager = EvaluationManager(config)
        print_success_message("Evaluation manager initialized successfully")
        
        # Step 2: Load initial configuration from config file
        print_step_header("Loading Configuration", 2, 6)
        
        # Initialize all configuration values from config file
        logger.info("Loading configuration from config file...")
        
        # Model configuration
        model_path = model_path or config.config.model.name
        logger.info(f"Model from config: {model_path}")
        
        # Batch size configuration
        batch_size = batch_size or config.config.model.batch_size
        logger.info(f"Batch size from config: {batch_size}")
        
        # Save predictions configuration
        save_predictions = save_predictions if save_predictions is not None else config.config.evaluation.save_predictions
        logger.info(f"Save predictions from config: {save_predictions}")
        
        # Output path configuration
        if not output:
            output = Path(config.config.output_dir) / f"{config.config.experiment_name}_results.json"
        logger.info(f"Output path from config: {output}")
        
        # Dataset configuration
        dataset_name = dataset or config.config.dataset.dataset_id
        logger.info(f"Dataset from config: {dataset_name}")
        
        # Max samples configuration
        config_max_samples = getattr(config.config.dataset, 'max_samples', None)
        if config_max_samples:
            logger.info(f"Max samples from config: {config_max_samples}")
        
        # Step 3: Apply command line overrides
        logger.info("Applying command line overrides...")
        
        # Model path override
        if model_path != config.config.model.name:
            logger.info(f"Model path overridden by command line: {model_path}")
            config.config.model.name = model_path
        
        # Batch size override
        if batch_size != config.config.model.batch_size:
            logger.info(f"Batch size overridden by command line: {batch_size}")
            config.config.model.batch_size = batch_size
        
        # Save predictions override
        if save_predictions != config.config.evaluation.save_predictions:
            logger.info(f"Save predictions overridden by command line: {save_predictions}")
            config.config.evaluation.save_predictions = save_predictions
        
        # Output path override
        if output != Path(config.config.output_dir) / f"{config.config.experiment_name}_results.json":
            logger.info(f"Output path overridden by command line: {output}")
            config.config.output_dir = str(output.parent) if hasattr(output, 'parent') else str(output)
        
        # Step 3: Load dataset with overrides
        print_step_header("Loading Dataset", 3, 6)
        
        # Load dataset based on configuration
        if dataset_name:
            samples = eval_manager.dataset_manager.load_dataset()
        else:
            raise ValueError("No dataset provided")
        
        # Apply max_samples limit
        if max_samples is not None:
            # Command line max_samples overrides config
            logger.info(f"Max samples overridden by command line: {max_samples}")
            if len(samples) > max_samples:
                samples = samples[:max_samples]
                logger.info(f"Limited evaluation to {max_samples} samples")
        elif config_max_samples and len(samples) > config_max_samples:
            # Use max_samples from config
            logger.info(f"Applying max samples from config: {config_max_samples}")
            samples = samples[:config_max_samples]
            logger.info(f"Limited evaluation to {config_max_samples} samples from config")
        
        # Print final dataset info
        print_dataset_info(
            dataset_name=dataset_name,
            sample_count=len(samples),
            source=config.config.dataset.source
        )
                
        # Step 4: Print model info
        print_step_header("Model Information", 4, 6)
        print_model_info(
            model_name=model_path.split("/")[-1] if "/" in model_path else model_path,
            model_path=model_path,
            device=config.config.model.device
        )
        
        # Step 5: Run evaluation
        print_step_header("Running Evaluation", 5, 6)
        logger.info(f"Starting evaluation of {len(samples)} samples")
        results = eval_manager.evaluate_model(
            model_path=model_path,
            samples=samples
        )
        
        # Step 6: Save and display results
        print_step_header("Saving Results", 6, 6)
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


if __name__ == '__main__':
    cli()
