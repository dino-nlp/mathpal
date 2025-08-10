"""Advanced usage example with custom configurations."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_pipeline.config import TrainingConfig, CometConfig
from training_pipeline.data import DatasetProcessor, ChatFormatter
from training_pipeline.models import ModelLoader, LoRAConfigManager, ModelSaver
from training_pipeline.training import TrainerFactory, TrainingUtils
from training_pipeline.experiments import CometTracker
from training_pipeline.inference import InferenceEngine
from training_pipeline.utils import setup_logging, DeviceUtils
from training_pipeline.utils.logging import TrainingLogger


def main():
    """Advanced usage example with custom configurations."""
    
    # Setup advanced logging
    setup_logging(log_level="INFO", log_file="examples/logs/advanced_training.log")
    logger = TrainingLogger("advanced_example")
    
    logger.separator("ADVANCED USAGE EXAMPLE")
    logger.info("Starting advanced Gemma3N training pipeline")
    
    # Print comprehensive device info
    DeviceUtils.print_device_info()
    
    # 1. Create advanced configurations
    logger.separator("CONFIGURATION")
    
    # Custom training config
    training_config = TrainingConfig(
        model_name="unsloth/gemma-3n-E4B-it",
        dataset_name="ngohongthai/exam-sixth_grade-instruct-dataset",
        max_seq_length=1024,  # Smaller for demo
        max_steps=50,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=25,
        experiment_name="advanced_example",
        output_dir="outputs/advanced_example",
        report_to="comet_ml"
    )
    
    # Custom LoRA config
    lora_config = LoRAConfigManager.create_lora_config(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        model_name=training_config.model_name
    )
    
    # Custom Comet config
    comet_config = CometConfig(
        experiment_name="advanced_example",
        tags=["gemma3n", "advanced", "demo", "vietnamese-math"]
    )
    
    logger.config("Training", training_config.to_dict())
    logger.config("LoRA", lora_config)
    
    # Validate configurations
    training_config.validate()
    
    # 2. Setup experiment tracking
    logger.separator("EXPERIMENT TRACKING")
    
    comet_tracker = None
    if training_config.report_to == "comet_ml":
        try:
            comet_tracker = CometTracker(comet_config)
            experiment = comet_tracker.setup_experiment(
                training_config=training_config.to_dict(),
                model_config=lora_config
            )
            if experiment:
                logger.success(f"Comet experiment: {comet_tracker.get_experiment_url()}")
        except Exception as e:
            logger.warning(f"Comet ML setup failed: {e}")
    
    # 3. Load model with custom LoRA config
    logger.separator("MODEL LOADING")
    
    model_loader = ModelLoader(training_config)
    logger.info("Loading base model...")
    model, tokenizer = model_loader.load_model_and_processor()
    
    logger.info("Applying custom LoRA configuration...")
    model = model_loader.apply_lora(model, lora_config)
    model_loader.print_model_info(model)
    
    # Log memory usage after model loading
    memory_info = DeviceUtils.get_cuda_memory_info()
    if "error" not in memory_info:
        logger.info(f"GPU memory after model loading: {memory_info['allocated_memory_gb']:.2f}GB")
    
    # 4. Advanced dataset processing
    logger.separator("DATASET PROCESSING")
    
    dataset_processor = DatasetProcessor(tokenizer)
    
    # Load both train and eval datasets
    datasets = dataset_processor.prepare_datasets(
        dataset_name=training_config.dataset_name,
        train_split=training_config.train_split,
        eval_split="test"  # Add eval split
    )
    
    train_dataset = datasets["train"]
    eval_dataset = datasets.get("eval")
    
    # Get dataset statistics
    train_stats = dataset_processor.get_dataset_stats(train_dataset)
    logger.dataset_info(
        dataset_name=training_config.dataset_name,
        train_size=train_stats["num_samples"],
        eval_size=len(eval_dataset) if eval_dataset else None
    )
    
    # Preview dataset
    dataset_processor.preview_dataset(train_dataset, num_samples=2)
    
    # 5. Create trainer with custom configuration
    logger.separator("TRAINER SETUP")
    
    trainer_factory = TrainerFactory(training_config)
    
    # Create training pipeline with eval dataset
    pipeline = trainer_factory.create_training_pipeline(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer = pipeline["trainer"]
    trainer_factory.print_training_info(trainer, train_dataset, eval_dataset)
    
    # 6. Monitor training with custom callbacks
    logger.separator("TRAINING")
    
    # Log training estimates
    TrainingUtils.print_training_estimates(
        num_samples=len(train_dataset),
        batch_size=training_config.get_effective_batch_size(),
        max_steps=training_config.max_steps,
        estimated_seconds_per_step=2.0  # Estimate
    )
    
    logger.training_start()
    start_time = TrainingUtils.time.time()
    
    # Start training
    trainer_stats = trainer.train()
    
    end_time = TrainingUtils.time.time()
    training_duration = end_time - start_time
    logger.training_complete(training_duration)
    
    
    # 7. Advanced inference testing
    # Test with different generation configs
    logger.separator("INFERENCE TESTING")
        
    inference_engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.8,
        top_p=0.9
    )
    
    # Test with different generation configs
    generation_configs = InferenceEngine.get_recommended_configs()
    
    test_question = "Tính diện tích hình chữ nhật có chiều dài 10m và chiều rộng 6m?"
    
    logger.info(f"Testing question: {test_question}")
    
    for config_name, config in generation_configs.items():
        logger.info(f"Testing with {config_name} config...")
        response = inference_engine.generate(
            question=test_question,
            generation_config=config
        )
        logger.info(f"  Response: {response[:100]}...")
    
    # Benchmark inference
    benchmark_questions = [
        "Tính 25 + 37 = ?",
        "Tìm chu vi hình vuông cạnh 8cm?",
        "Giải phương trình: x + 15 = 32"
    ]
    
    logger.info("Running inference benchmark...")
    benchmark_results = inference_engine.benchmark_inference(
        questions=benchmark_questions,
        num_runs=2
    )
    
    logger.metric("Average tokens/second", benchmark_results["avg_tokens_per_second"])
    
    # 8. Advanced model saving
    logger.separator("MODEL SAVING")
    
    model_saver = ModelSaver(model, tokenizer)
    
    # Save multiple formats
    save_formats = {
        "lora": {},
        "merged_fp16": {"precision": "fp16"},
        "gguf_q8": {"quantization": "q8_0"}
    }
    
    logger.info("Saving model in multiple formats...")
    save_results = model_saver.save_all_formats(
        base_save_path=training_config.get_output_dir(),
        model_name="gemma3n-advanced",
        formats=save_formats
    )
    
    for format_name, save_path in save_results.items():
        if not save_path.startswith("Error"):
            logger.save_model(save_path, format_name)
        else:
            logger.error(f"Failed to save {format_name}: {save_path}")
    
    # 9. Log comprehensive results
    if comet_tracker and comet_tracker.is_active():
        logger.separator("EXPERIMENT LOGGING")
        
        # Log training summary
        comet_tracker.log_training_summary(
            training_stats=trainer_stats.__dict__,
            final_metrics={
                "training_duration": training_duration,
                "avg_tokens_per_second": benchmark_results["avg_tokens_per_second"],
                "final_train_loss": getattr(trainer_stats, "train_loss", 0.0)
            }
        )
        
        # Log models
        for format_name, save_path in save_results.items():
            if not save_path.startswith("Error"):
                comet_tracker.log_model(save_path, f"model_{format_name}")
        
        # Log additional info
        comet_tracker.log_parameter("lora_rank", lora_config["r"])
        comet_tracker.log_parameter("effective_batch_size", training_config.get_effective_batch_size())
        
        logger.success(f"Experiment logged: {comet_tracker.get_experiment_url()}")
        comet_tracker.end_experiment()
    
    # 10. Create training summary
    logger.separator("SUMMARY")
    
    summary = TrainingUtils.create_training_summary(
        trainer_stats=trainer_stats,
        config=training_config.to_dict(),
        save_path=f"{training_config.get_output_dir()}/training_summary.json"
    )
    
    logger.success("Advanced training pipeline completed successfully!")
    logger.info(f"Output directory: {training_config.get_output_dir()}")
    logger.info(f"Training duration: {training_duration:.2f} seconds")
    logger.info(f"Average inference speed: {benchmark_results['avg_tokens_per_second']:.1f} tokens/second")
    
    # Final cleanup
    DeviceUtils.clear_cuda_cache()
    logger.info("Memory cache cleared")
    
    logger.separator("COMPLETED")


if __name__ == "__main__":
    main()