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

from src.evaluation_pipeline.factories.model_factory import ModelFactory
from src.evaluation_pipeline.inference.inference_engine import InferenceEngine

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
@click.pass_context
def evaluate(ctx):
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
    from src.evaluation_pipeline.managers.dataset_manager import DatasetManager
    
    config = ctx.obj['config'] # ConfigManager
    logger = ctx.obj['logger']
    
    # Print evaluation header
    print_evaluation_header()
    try:
        # Step 1: Load initial configuration from config file
        print_step_header("Loading Configuration", 1, 6)
        logger.info(config.summary())
        # Step 2: Create evaluation manager
        print_step_header("Initializing Evaluation Manager", 2, 6)
        eval_manager = EvaluationManager(config)
        print_success_message("Evaluation manager initialized successfully")
        
        # Step 3: Load dataset with overrides
        print_step_header("Loading Dataset", 3, 6)
        
        # Load dataset based on configuration
        dataset_manager = DatasetManager(config)
        evaluation_samples = dataset_manager.load_dataset()
        
        # Print final dataset info
        print_dataset_info(
            dataset_name=config.get_dataset_config().dataset_id,
            sample_count=len(evaluation_samples)
        )
                
        # Step 4: Load model
        print_step_header("Loading model", 4, 6)
        model_factory = ModelFactory(config)
        model, tokenizer = model_factory.load_model()

        
        # Step 5: Generate response
        print_step_header("Generating Response", 5, 6)
        inference_engine = InferenceEngine(model, tokenizer, config, device=config.get_model_config().device)
        response = inference_engine.generate(evaluation_samples[0]['question'])
        print_success_message("Response generated successfully")
        print(response)
        
        print_step_header("Running Evaluation", 5, 6)
        logger.info(f"Starting evaluation of {len(samples)} samples")
        results = eval_manager.evaluate_model(
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
