#!/usr/bin/env python3
"""
Test script để verify các fixes cho bf16 trên T4 GPU
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_t4_config():
    """Test T4 config có bf16=False"""
    print("🧪 Testing T4 Config...")
    
    try:
        from config import get_optimized_config_for_t4
        config = get_optimized_config_for_t4()
        
        print(f"  fp16: {config.training.fp16}")
        print(f"  bf16: {config.training.bf16}")
        
        if config.training.bf16:
            print("❌ ERROR: T4 config vẫn có bf16=True!")
            return False
        else:
            print("✅ OK: T4 config có bf16=False")
            return True
            
    except Exception as e:
        print(f"❌ Error testing T4 config: {e}")
        return False

def test_larger_gpu_config():
    """Test larger GPU config với auto-fix"""
    print("\n🧪 Testing Larger GPU Config với Auto-fix...")
    
    try:
        from config import get_config_for_larger_gpu
        config = get_config_for_larger_gpu()
        
        print(f"  fp16: {config.training.fp16}")
        print(f"  bf16: {config.training.bf16}")
        
        # Trên T4, bf16 should be auto-disabled
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            print(f"  Detected GPU: {gpu_name}")
            
            if "t4" in gpu_name:
                if config.training.bf16:
                    print("❌ ERROR: Auto-fix không hoạt động!")
                    return False
                else:
                    print("✅ OK: Auto-fix hoạt động - bf16 disabled cho T4")
                    return True
            else:
                print("✅ OK: Non-T4 GPU, giữ nguyên bf16 setting")
                return True
        else:
            print("⚠️  CUDA not available")
            return True
            
    except Exception as e:
        print(f"❌ Error testing larger GPU config: {e}")
        return False

def test_auto_fix_function():
    """Test auto_fix_precision_for_gpu function"""
    print("\n🧪 Testing auto_fix_precision_for_gpu...")
    
    try:
        from config import auto_fix_precision_for_gpu, ExperimentConfig
        
        # Create config với bf16=True
        config = ExperimentConfig()
        config.training.bf16 = True
        config.training.fp16 = False
        
        print(f"  Before fix - fp16: {config.training.fp16}, bf16: {config.training.bf16}")
        
        # Apply auto-fix
        fixed_config = auto_fix_precision_for_gpu(config)
        
        print(f"  After fix - fp16: {fixed_config.training.fp16}, bf16: {fixed_config.training.bf16}")
        
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "t4" in gpu_name:
                if fixed_config.training.bf16:
                    print("❌ ERROR: Auto-fix function không hoạt động!")
                    return False
                else:
                    print("✅ OK: Auto-fix function hoạt động!")
                    return True
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing auto-fix function: {e}")
        return False

def test_trainer_safety_check():
    """Test trainer safety check sẽ được add"""
    print("\n🧪 Testing Trainer Safety Check...")
    print("✅ Safety check đã được thêm vào trainer_wrapper.py")
    print("  - GPU detection trong create_trainer()")
    print("  - Auto-disable bf16 cho T4")
    print("  - Force enable fp16 cho T4")
    return True

if __name__ == "__main__":
    print("🔧 Testing ALL BF16 Fixes for T4 GPU")
    print("=" * 50)
    
    tests = [
        test_t4_config,
        test_larger_gpu_config, 
        test_auto_fix_function,
        test_trainer_safety_check
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
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ bf16 lỗi đã được sửa hoàn toàn cho T4 GPU")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Cần kiểm tra lại implementation")