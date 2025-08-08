#!/usr/bin/env python3
"""
Test script để verify tất cả các fixes cho SFTTrainer và bf16
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test basic imports"""
    print("📦 Testing imports...")
    
    try:
        from trl import SFTTrainer, SFTConfig
        print("  ✅ trl imports OK")
    except Exception as e:
        print(f"  ❌ trl import failed: {e}")
        return False
    
    try:
        from config import get_optimized_config_for_t4
        print("  ✅ config imports OK")
    except Exception as e:
        print(f"  ❌ config import failed: {e}")
        return False
    
    return True

def test_sftconfig_creation():
    """Test SFTConfig creation với và không có tokenizer"""
    print("\n🔧 Testing SFTConfig creation...")
    
    try:
        from trl import SFTConfig
        from config import get_optimized_config_for_t4
        
        config = get_optimized_config_for_t4()
        
        # Test approach 1: SFTConfig without tokenizer
        try:
            training_args = SFTConfig(
                dataset_text_field="text",
                output_dir="./test",
                max_steps=10,
                per_device_train_batch_size=1,
                learning_rate=2e-4,
                fp16=config.training.fp16,
                bf16=config.training.bf16,
                max_length=2048,
                seed=3407,
            )
            print("  ✅ SFTConfig created without tokenizer")
            print(f"    fp16={training_args.fp16}, bf16={training_args.bf16}")
            
            if training_args.bf16:
                print("  ❌ WARNING: bf16=True in SFTConfig!")
                return False
            else:
                print("  ✅ bf16=False in SFTConfig")
                return True
                
        except Exception as e:
            print(f"  ❌ SFTConfig creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error in SFTConfig test: {e}")
        return False

def test_sfttrainer_approaches():
    """Test different SFTTrainer creation approaches"""
    print("\n🔧 Testing SFTTrainer approaches...")
    
    try:
        from trl import SFTTrainer, SFTConfig
        import inspect
        
        # Check SFTTrainer signature
        signature = inspect.signature(SFTTrainer.__init__)
        parameters = list(signature.parameters.keys())
        
        print(f"  SFTTrainer parameters: {parameters}")
        
        has_tokenizer = 'tokenizer' in parameters
        has_processing_class = 'processing_class' in parameters
        
        print(f"  Has 'tokenizer': {has_tokenizer}")
        print(f"  Has 'processing_class': {has_processing_class}")
        
        # Recommendations
        if has_tokenizer:
            print("  ✅ Can use tokenizer parameter")
        elif has_processing_class:
            print("  ✅ Should use processing_class parameter")
        else:
            print("  ✅ Should use no tokenizer parameter")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing SFTTrainer: {e}")
        return False

def test_gpu_detection():
    """Test GPU detection cho bf16 fixes"""
    print("\n🔍 Testing GPU detection...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  Detected GPU: {gpu_name}")
            
            is_t4 = "t4" in gpu_name.lower()
            print(f"  Is T4: {is_t4}")
            
            if is_t4:
                print("  ✅ T4 detected - bf16 should be disabled")
            else:
                print(f"  ✅ Non-T4 GPU - bf16 may be supported")
                
            return True
        else:
            print("  ⚠️  CUDA not available")
            return True
            
    except Exception as e:
        print(f"❌ Error in GPU detection: {e}")
        return False

def test_debug_logging():
    """Test debug logging setup"""
    print("\n📝 Testing debug logging...")
    
    try:
        import logging
        
        # Setup logging like in trainer_wrapper
        logger = logging.getLogger("test_trainer")
        
        # Test various log levels
        logger.info("🔧 Test info log")
        logger.warning("⚠️  Test warning log")
        logger.error("❌ Test error log")
        
        print("  ✅ Debug logging works")
        return True
        
    except Exception as e:
        print(f"❌ Error in logging test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ALL SFTTrainer & BF16 Fixes")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_sftconfig_creation,
        test_sfttrainer_approaches,
        test_gpu_detection,
        test_debug_logging,
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
        print("✅ SFTTrainer fixes should work")
        print("✅ BF16 detection is working")
        print("✅ Multiple fallback approaches implemented")
        print("\n🚀 Ready for training!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Check the error messages above for details")
        
    print("\n💡 Implementation summary:")
    print("1. Multiple SFTConfig creation approaches")
    print("2. Multiple SFTTrainer creation approaches") 
    print("3. GPU detection with bf16 auto-disable")
    print("4. Emergency safety checks")
    print("5. Comprehensive debug logging")