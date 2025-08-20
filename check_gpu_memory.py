#!/usr/bin/env python3
"""
Script to check GPU memory and provide recommendations for model loading.
"""

import torch
import psutil
import os

def check_gpu_memory():
    """Check GPU memory availability"""
    print("🔍 GPU Memory Check")
    print("=" * 40)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3  # GB
            free_memory = total_memory - allocated_memory
            
            print(f"\n📊 GPU {i}: {gpu_name}")
            print(f"   Total Memory: {total_memory:.2f} GB")
            print(f"   Allocated Memory: {allocated_memory:.2f} GB")
            print(f"   Cached Memory: {cached_memory:.2f} GB")
            print(f"   Free Memory: {free_memory:.2f} GB")
            
            # Recommendations
            if free_memory < 8:
                print(f"   ⚠️  Low memory available ({free_memory:.2f} GB)")
                print(f"   💡 Consider using CPU offload or 8-bit quantization")
            elif free_memory < 16:
                print(f"   ⚠️  Moderate memory available ({free_memory:.2f} GB)")
                print(f"   💡 4-bit quantization should work")
            else:
                print(f"   ✅ Sufficient memory available ({free_memory:.2f} GB)")
                print(f"   💡 Standard loading should work fine")
    else:
        print("❌ CUDA not available")
        print("💡 Model will be loaded on CPU (slower but works)")

def check_system_memory():
    """Check system memory"""
    print("\n🔍 System Memory Check")
    print("=" * 40)
    
    memory = psutil.virtual_memory()
    total_gb = memory.total / 1024**3
    available_gb = memory.available / 1024**3
    used_gb = memory.used / 1024**3
    
    print(f"📊 Total RAM: {total_gb:.2f} GB")
    print(f"📊 Used RAM: {used_gb:.2f} GB")
    print(f"📊 Available RAM: {available_gb:.2f} GB")
    print(f"📊 Usage: {memory.percent:.1f}%")
    
    if available_gb < 8:
        print("⚠️  Low system memory available")
        print("💡 Consider closing other applications")
    elif available_gb < 16:
        print("⚠️  Moderate system memory available")
        print("💡 CPU offload may be slow")
    else:
        print("✅ Sufficient system memory available")

def check_environment():
    """Check environment variables and settings"""
    print("\n🔍 Environment Check")
    print("=" * 40)
    
    # Check CUDA environment
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"📊 CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Check PyTorch version
    print(f"📊 PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    print(f"📊 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"📊 CUDA version: {torch.version.cuda}")

def provide_recommendations():
    """Provide recommendations based on system state"""
    print("\n💡 Recommendations")
    print("=" * 40)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_free = 0
        
        for i in range(gpu_count):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            free_memory = total_memory - allocated_memory
            total_free += free_memory
        
        if total_free < 8:
            print("🚨 Low GPU memory detected!")
            print("   Recommended actions:")
            print("   1. Use CPU offload: --device_map auto")
            print("   2. Use 8-bit quantization: load_in_8bit=True")
            print("   3. Reduce batch size or sequence length")
            print("   4. Close other GPU applications")
            
        elif total_free < 16:
            print("⚠️  Moderate GPU memory available")
            print("   Recommended actions:")
            print("   1. Use 4-bit quantization: load_in_4bit=True")
            print("   2. Monitor memory usage during evaluation")
            print("   3. Consider using smaller model if available")
            
        else:
            print("✅ Sufficient GPU memory available")
            print("   Standard model loading should work fine")
            
    else:
        print("💻 CPU-only mode detected")
        print("   Recommended actions:")
        print("   1. Use CPU offload: device_map='auto'")
        print("   2. Be patient - CPU inference is slower")
        print("   3. Consider using smaller model")
        print("   4. Ensure sufficient system RAM")

def main():
    """Run all checks"""
    print("🔍 MathPal GPU Memory Checker")
    print("=" * 50)
    
    check_gpu_memory()
    check_system_memory()
    check_environment()
    provide_recommendations()
    
    print("\n" + "=" * 50)
    print("🎯 Next steps:")
    print("   1. If GPU memory is low, try: make evaluate-llm-fast")
    print("   2. For quick testing: make evaluate-llm-quick")
    print("   3. Check logs for detailed memory usage")
    print("   4. Consider using smaller model if available")

if __name__ == "__main__":
    main()
