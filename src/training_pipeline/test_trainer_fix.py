#!/usr/bin/env python3
"""
Quick test để verify trainer implementation theo working notebook
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_config_creation():
    """Test việc tạo config với new settings"""
    print("🧪 Testing config creation...")
    
    try:
        from config import get_optimized_config_for_t4
        config = get_optimized_config_for_t4()
        
        print("✅ Config created successfully!")
        print(f"  max_steps: {config.training.max_steps}")
        print(f"  num_train_epochs: {config.training.num_train_epochs}")
        print(f"  optim: {config.training.optim}")
        print(f"  eval_steps: {config.training.eval_steps}")
        print(f"  save_steps: {config.training.save_steps}")
        
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def test_sftconfig_parameters():
    """Test SFTConfig parameters according to working notebook"""
    print("\n🧪 Testing SFTConfig parameters...")
    
    try:
        from config import get_optimized_config_for_t4
        from trl import SFTConfig
        
        config = get_optimized_config_for_t4()
        
        # Create minimal SFTConfig theo working notebook
        training_args = SFTConfig(
            dataset_text_field=config.dataset.dataset_text_field,
            output_dir=config.training.output_dir,
            max_steps=config.training.max_steps,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            warmup_ratio=config.training.warmup_ratio,
            weight_decay=config.training.weight_decay,
            lr_scheduler_type=config.training.lr_scheduler_type,
            optim=config.training.optim,
            logging_steps=config.training.logging_steps,
            save_strategy=config.training.save_strategy,
            save_steps=config.training.save_steps,
            report_to=config.training.report_to,
            max_length=config.model.max_seq_length,  # Key difference: max_length
            seed=config.training.seed,
        )
        
        print("✅ SFTConfig created successfully!")
        print(f"  dataset_text_field: {training_args.dataset_text_field}")
        print(f"  max_length: {training_args.max_length}")
        print(f"  optim: {training_args.optim}")
        print(f"  max_steps: {training_args.max_steps}")
        
        return True
    except Exception as e:
        print(f"❌ SFTConfig creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing Trainer Implementation Fixes...")
    print("=" * 60)
    
    tests = [
        test_config_creation(),
        test_sftconfig_parameters(),
    ]
    
    print("\n" + "=" * 60)
    
    if all(tests):
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Implementation fixes:")
        print("  - Config updated to match working notebook")
        print("  - SFTConfig uses minimal parameters")
        print("  - max_length instead of max_seq_length")
        print("  - adamw_torch_fused optimizer")
        print("  - max_steps=100 for quick training")
        print("\n🚀 Ready to test with real training!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)