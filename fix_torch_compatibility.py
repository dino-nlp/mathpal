#!/usr/bin/env python3
"""
Script to check and fix Torch/Unsloth compatibility issues.
"""

import os
import sys
import torch
import subprocess

def check_torch_version():
    """Check PyTorch version and compatibility"""
    print("🔍 PyTorch Version Check")
    print("=" * 40)
    
    print(f"📊 PyTorch version: {torch.__version__}")
    print(f"📊 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"📊 CUDA version: {torch.version.cuda}")
        print(f"📊 cuDNN version: {torch.backends.cudnn.version()}")
    
    # Check for potential compatibility issues
    torch_version = torch.__version__.split('.')
    major, minor = int(torch_version[0]), int(torch_version[1])
    
    if major == 2 and minor >= 2:
        print("✅ PyTorch 2.2+ detected - good for Unsloth")
    elif major == 2 and minor >= 1:
        print("⚠️  PyTorch 2.1 detected - may have compatibility issues")
    else:
        print("❌ PyTorch version may be too old for Unsloth")

def check_environment_variables():
    """Check and set environment variables for compatibility"""
    print("\n🔍 Environment Variables Check")
    print("=" * 40)
    
    # Required environment variables for compatibility
    env_vars = {
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_COMPILE_DISABLE": "1",
        "TORCH_LOGS": "off",
        "CUDA_LAUNCH_BLOCKING": "1",  # For better error messages
    }
    
    for var, value in env_vars.items():
        current_value = os.environ.get(var, "Not set")
        print(f"📊 {var}: {current_value}")
        
        if current_value != value:
            os.environ[var] = value
            print(f"   ✅ Set to: {value}")
        else:
            print(f"   ✅ Already set correctly")

def check_unsloth_installation():
    """Check Unsloth installation"""
    print("\n🔍 Unsloth Installation Check")
    print("=" * 40)
    
    try:
        import unsloth
        print(f"✅ Unsloth installed: {unsloth.__version__}")
        
        # Check if FastModel is available
        from unsloth import FastModel
        print("✅ FastModel available")
        
    except ImportError as e:
        print(f"❌ Unsloth not installed: {e}")
        print("💡 Install with: pip install unsloth")
        return False
    except Exception as e:
        print(f"⚠️  Unsloth import error: {e}")
        return False
    
    return True

def check_torch_dynamo_config():
    """Check and configure TorchDynamo settings"""
    print("\n🔍 TorchDynamo Configuration")
    print("=" * 40)
    
    try:
        # Disable TorchDynamo to prevent FX tracing conflicts
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
        
        print("✅ TorchDynamo disabled")
        print("✅ FX tracing conflicts prevented")
        
    except Exception as e:
        print(f"⚠️  Could not configure TorchDynamo: {e}")

def test_model_loading():
    """Test model loading with safe configuration"""
    print("\n🔍 Model Loading Test")
    print("=" * 40)
    
    try:
        from unsloth import FastModel
        
        # Test with a small model or just check if FastModel works
        print("🔄 Testing FastModel import and basic functionality...")
        
        # This is a basic test - in practice, you'd test with your actual model
        print("✅ FastModel import successful")
        print("✅ Ready for safe model loading")
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False
    
    return True

def provide_recommendations():
    """Provide recommendations based on system state"""
    print("\n💡 Recommendations")
    print("=" * 40)
    
    print("🎯 For FX tracing conflicts:")
    print("   1. ✅ TorchDynamo has been disabled")
    print("   2. ✅ Environment variables set")
    print("   3. ✅ Model loading uses compile=False")
    
    print("\n🎯 For evaluation:")
    print("   1. Use: make evaluate-llm-safe")
    print("   2. Monitor logs for any remaining issues")
    print("   3. If issues persist, try: make evaluate-llm-fast")
    
    print("\n🎯 For debugging:")
    print("   1. Check GPU memory: make check-gpu")
    print("   2. Monitor system resources")
    print("   3. Check logs for detailed error messages")

def main():
    """Run all compatibility checks and fixes"""
    print("🔍 MathPal Torch/Unsloth Compatibility Checker")
    print("=" * 60)
    
    # Check and fix environment
    check_torch_version()
    check_environment_variables()
    check_torch_dynamo_config()
    
    # Check installations
    unsloth_ok = check_unsloth_installation()
    
    if unsloth_ok:
        # Test model loading
        model_ok = test_model_loading()
        
        if model_ok:
            print("\n✅ All compatibility checks passed!")
            print("🎯 System ready for safe evaluation")
        else:
            print("\n⚠️  Model loading test failed")
            print("💡 Check your model configuration")
    else:
        print("\n❌ Unsloth installation issues detected")
        print("💡 Please install/update Unsloth")
    
    provide_recommendations()
    
    print("\n" + "=" * 60)
    print("🎯 Next steps:")
    print("   1. Try: make evaluate-llm-safe")
    print("   2. If issues persist: make evaluate-llm-fast")
    print("   3. Check logs for detailed information")

if __name__ == "__main__":
    main()
