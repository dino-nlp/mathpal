"""Basic usage example for Gemma3N training pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_pipeline.config import TrainingConfig, CometConfig
from training_pipeline.data import DatasetProcessor
from training_pipeline.models import ModelLoader, ModelSaver
from training_pipeline.training import TrainerFactory
from training_pipeline.experiments import CometTracker
from training_pipeline.inference import InferenceEngine
from training_pipeline.utils import setup_logging, DeviceUtils


def main():
    """Basic usage example."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    print("🚀 Gemma3N Training Pipeline - Basic Usage Example")
    
    # Print device info
    DeviceUtils.print_device_info()
    
    # 1. Create configurations
    print("\n📋 Step 1: Creating configurations...")
    
    training_config = TrainingConfig(
        model_name="unsloth/gemma-3n-E4B-it",
        dataset_name="ngohongthai/exam-sixth_grade-instruct-dataset",
        max_steps=20,  # Quick test
        experiment_name="basic_example",
        output_dir="outputs/basic_example"
    )
    
    comet_config = CometConfig(experiment_name="basic_example")
    
    print(f"✅ Training config created - Max steps: {training_config.max_steps}")
    
    # 2. Setup experiment tracking (optional)
    print("\n📊 Step 2: Setting up experiment tracking...")
    
    comet_tracker = None
    if training_config.report_to == "comet_ml":
        try:
            comet_tracker = CometTracker(comet_config)
            comet_tracker.setup_experiment(training_config.to_dict())
        except Exception as e:
            print(f"⚠️ Comet ML setup failed: {e}")
            print("Continuing without experiment tracking...")
    
    # 3. Load model and tokenizer
    print("\n🤖 Step 3: Loading model and tokenizer...")
    
    model_loader = ModelLoader(training_config)
    model, tokenizer = model_loader.load_complete_model(apply_lora=True)
    model_loader.print_model_info(model)
    
    # 4. Load and prepare dataset
    print("\n📊 Step 4: Loading and preparing dataset...")
    
    dataset_processor = DatasetProcessor(tokenizer)
    train_dataset = dataset_processor.prepare_dataset(
        dataset_name=training_config.dataset_name,
        split=training_config.train_split
    )
    
    dataset_processor.preview_dataset(train_dataset, num_samples=1)
    
    # 5. Create trainer
    print("\n🏋️ Step 5: Creating trainer...")
    
    trainer_factory = TrainerFactory(training_config)
    trainer = trainer_factory.create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset
    )
    
    trainer_factory.print_training_info(trainer, train_dataset)
    
    # 6. Start training
    print("\n🚀 Step 6: Starting training...")
    
    trainer_stats = trainer.train()
    print("✅ Training completed!")
    
    # 7. Test inference
    print("\n🧪 Step 7: Testing inference...")
    
    inference_engine = InferenceEngine(model, tokenizer, max_new_tokens=32)
    
    test_questions = [
        "Tính tổng của 15 + 27 = ?",
        "Một hình chữ nhật có chiều dài 8m và chiều rộng 5m. Tính diện tích?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}:")
        print(f"Q: {question}")
        answer = inference_engine.generate(question)
        print(f"A: {answer}")
    
    # 8. Save model (LoRA adapters only for quick example)
    print("\n💾 Step 8: Saving model...")
    
    model_saver = ModelSaver(model, tokenizer)
    save_path = model_saver.save_lora_adapters(
        save_path=f"{training_config.get_output_dir()}/lora_adapters"
    )
    print(f"✅ Model saved to: {save_path}")
    
    # 9. Log results to experiment tracker
    if comet_tracker and comet_tracker.is_active():
        print("\n📈 Step 9: Logging results...")
        comet_tracker.log_training_summary(trainer_stats.__dict__)
        comet_tracker.log_model(save_path, "lora_adapters")
        comet_tracker.end_experiment()
    
    print("\n🎉 Basic usage example completed successfully!")
    print(f"📁 Output directory: {training_config.get_output_dir()}")
    
    # Cleanup
    DeviceUtils.clear_cuda_cache()


if __name__ == "__main__":
    main()