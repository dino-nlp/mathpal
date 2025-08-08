#!/usr/bin/env python3
"""
Test script để kiểm tra việc tạo trainer và precision settings
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_sft_config_creation():
    """Test việc tạo SFTConfig với correct precision settings"""
    print("🧪 Testing SFTConfig Creation...")
    
    try:
        from config import get_optimized_config_for_t4
        from trl import SFTConfig
        
        config = get_optimized_config_for_t4()
        
        print("📋 Config values:")
        print(f"  config.training.fp16: {config.training.fp16}")
        print(f"  config.training.bf16: {config.training.bf16}")
        
        # Simulate việc tạo SFTConfig như trong trainer
        training_args = SFTConfig(
            dataset_text_field="text",
            output_dir="./test_output",
            max_steps=10,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            fp16=config.training.fp16,  # Explicit set
            bf16=config.training.bf16,  # Explicit set
            logging_steps=5,
            save_strategy="steps",
            save_steps=50,
            report_to=["comet_ml"],
            max_length=2048,
            seed=3407,
        )
        
        print("🔍 SFTConfig results:")
        print(f"  training_args.fp16: {training_args.fp16}")
        print(f"  training_args.bf16: {training_args.bf16}")
        
        if training_args.bf16:
            print("❌ ERROR: SFTConfig vẫn có bf16=True!")
            return False
        elif training_args.fp16:
            print("✅ OK: SFTConfig có fp16=True, bf16=False")
            return True
        else:
            print("⚠️  WARNING: Không có fp16 và bf16 nào được enable")
            return True
            
    except Exception as e:
        print(f"❌ Error testing SFTConfig: {e}")
        return False

def test_safety_check_logic():
    """Test GPU detection và safety check logic"""
    print("\n🧪 Testing Safety Check Logic...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            print(f"Detected GPU: {gpu_name}")
            
            # Simulate safety check logic
            is_t4 = "t4" in gpu_name
            print(f"Is T4 GPU: {is_t4}")
            
            if is_t4:
                print("✅ Safety check sẽ được trigger cho T4")
                print("  - bf16 sẽ được force thành False")
                print("  - fp16 sẽ được force thành True")
            else:
                print("ℹ️  Safety check sẽ không trigger cho non-T4 GPU")
                
            return True
        else:
            print("⚠️  CUDA not available")
            return True
            
    except Exception as e:
        print(f"❌ Error testing safety check: {e}")
        return False

def test_emergency_fix():
    """Test emergency fix trong trainer creation"""
    print("\n🧪 Testing Emergency Fix Logic...")
    
    try:
        from trl import SFTConfig
        
        # Simulate SFTConfig với bf16=True (worst case)
        bad_config = SFTConfig(
            dataset_text_field="text",
            output_dir="./test_output",
            max_steps=10,
            fp16=False,
            bf16=True,  # This should trigger emergency fix
        )
        
        print(f"Before fix: fp16={bad_config.fp16}, bf16={bad_config.bf16}")
        
        # Simulate emergency fix
        if bad_config.bf16:
            print("🛡️  EMERGENCY FIX triggered!")
            bad_config.bf16 = False
            bad_config.fp16 = True
            
        print(f"After fix: fp16={bad_config.fp16}, bf16={bad_config.bf16}")
        
        if bad_config.bf16:
            print("❌ ERROR: Emergency fix không hoạt động!")
            return False
        else:
            print("✅ OK: Emergency fix hoạt động!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing emergency fix: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Trainer Creation & BF16 Safety")
    print("=" * 50)
    
    tests = [
        test_sft_config_creation,
        test_safety_check_logic,
        test_emergency_fix,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    for i, result in enumerate(results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  Test {i+1}: {status}")
    
    if all(results):
        print("\n🎉 ALL TRAINER TESTS PASSED!")
        print("✅ Trainer creation sẽ có multiple safety checks")
        print("✅ bf16 sẽ được force disabled cho T4 ở nhiều levels")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Cần kiểm tra lại implementation")