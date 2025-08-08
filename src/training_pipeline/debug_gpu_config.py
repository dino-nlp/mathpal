#!/usr/bin/env python3
"""
Debug script để kiểm tra GPU detection và config selection
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def debug_gpu_detection():
    """Debug GPU detection process"""
    print("🔍 Debug GPU Detection Process...")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")
                
                # Test detection logic
                gpu_name_lower = gpu_name.lower()
                print(f"  - Lowercase name: '{gpu_name_lower}'")
                print(f"  - Contains 't4': {'t4' in gpu_name_lower}")
                print(f"  - Contains 'a100': {'a100' in gpu_name_lower}")
                print(f"  - Contains 'v100': {'v100' in gpu_name_lower}")
                
            # Run actual detection
            from train_gemma3n import detect_gpu_type
            detected_type = detect_gpu_type()
            print(f"\n✅ Detected GPU type: '{detected_type}'")
        else:
            print("❌ CUDA not available")
            
    except Exception as e:
        print(f"❌ Error during GPU detection: {e}")

def debug_config_selection():
    """Debug config selection and check bf16/fp16 settings"""
    print("\n🔍 Debug Config Selection...")
    print("=" * 50)
    
    try:
        from train_gemma3n import detect_gpu_type
        from config import get_optimized_config_for_t4, get_config_for_larger_gpu
        
        gpu_type = detect_gpu_type()
        print(f"Detected GPU type: '{gpu_type}'")
        
        if gpu_type == "t4":
            print("📋 Using T4 config...")
            config = get_optimized_config_for_t4()
        else:
            print("📋 Using Larger GPU config...")
            config = get_config_for_larger_gpu()
        
        print(f"  fp16: {config.training.fp16}")
        print(f"  bf16: {config.training.bf16}")
        print(f"  optim: {config.training.optim}")
        print(f"  max_seq_length: {config.model.max_seq_length}")
        print(f"  batch_size: {config.training.per_device_train_batch_size}")
        
        # Validation
        if config.training.bf16 and gpu_type == "t4":
            print("❌ ERROR: bf16=True cho T4 GPU - này sẽ gây lỗi!")
            print("💡 Solution: Cần đảm bảo bf16=False cho T4")
        elif config.training.fp16 and gpu_type == "t4":
            print("✅ OK: fp16=True cho T4 GPU")
        else:
            print("⚠️  Warning: Cấu hình có thể không tối ưu")
            
    except Exception as e:
        print(f"❌ Error during config selection: {e}")

def debug_manual_config():
    """Test manual T4 config để đảm bảo đúng"""
    print("\n🔍 Debug Manual T4 Config...")
    print("=" * 50)
    
    try:
        from config import get_optimized_config_for_t4
        config = get_optimized_config_for_t4()
        
        print("Manual T4 config:")
        print(f"  fp16: {config.training.fp16}")
        print(f"  bf16: {config.training.bf16}")
        
        if config.training.bf16:
            print("❌ ERROR: Manual T4 config có bf16=True!")
        else:
            print("✅ OK: Manual T4 config có bf16=False")
            
    except Exception as e:
        print(f"❌ Error during manual config test: {e}")

if __name__ == "__main__":
    debug_gpu_detection()
    debug_config_selection()
    debug_manual_config()
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY:")
    print("1. Kiểm tra GPU detection có đúng không")
    print("2. Kiểm tra config selection logic")  
    print("3. Verify T4 config có bf16=False")
    print("4. Nếu vẫn lỗi, cần force bf16=False trong code")