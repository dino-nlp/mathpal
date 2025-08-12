#!/usr/bin/env python3
"""Setup script for the training environment."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str = "") -> bool:
    """Run a command and return success status."""
    if description:
        print(f"📦 {description}...")
    
    print(f"🔧 Running: {command}")
    
    try:
        # Check if command contains shell operators like pipes
        if "|" in command or ">" in command or "<" in command:
            # Use shell=True for complex shell commands
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
        else:
            # Use split for simple commands (safer)
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                check=True
            )
        
        print(f"✅ Success: {description or command}")
        if result.stdout:
            print(f"📄 Output: {result.stdout[:200]}...")  # Show first 200 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description or command}")
        print(f"💥 Return code: {e.returncode}")
        if e.stderr:
            print(f"🚨 Error output: {e.stderr}")
        if e.stdout:
            print(f"📄 Standard output: {e.stdout}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("🔥 Checking CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            print(f"✅ CUDA {cuda_version} available")
            print(f"   Devices: {device_count}")
            print(f"   Primary device: {device_name}")
            return True
        else:
            print("⚠️ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed - cannot check CUDA")
        return False


def check_current_installation():
    """Check current installation and fix issues."""
    print("🔍 Checking current installation...")
    
    try:
        import torch
        print(f"🔧 PyTorch version: {torch.__version__}")
        
        import transformers
        print(f"🔧 Transformers version: {transformers.__version__}")
        
        try:
            import unsloth
            print(f"🔧 Unsloth version: {unsloth.__version__ if hasattr(unsloth, '__version__') else 'unknown'}")
        except ImportError as e:
            print(f"⚠️ Unsloth import error: {e}")
            return False
            
        try:
            import unsloth_zoo
            print(f"🔧 Unsloth Zoo available")
        except ImportError as e:
            print(f"⚠️ Unsloth Zoo import error: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Installation check failed: {e}")
        return False


def fix_unsloth_installation():
    """Fix Unsloth installation issues."""
    print("🔧 Fixing Unsloth installation...")
    
    # First, try to uninstall problematic packages
    cleanup_commands = [
        ("pip uninstall -y unsloth unsloth_zoo xformers", "Cleaning up problematic packages"),
        ("pip install --upgrade pip", "Upgrading pip"),
    ]
    
    for command, description in cleanup_commands:
        run_command(command, description)
    
    # Install compatible transformers version first
    compatibility_commands = [
        ("pip install transformers==4.52.4", "Installing compatible Transformers"),
        ("pip install triton>=3.3.1", "Installing compatible Triton"),
    ]
    
    for command, description in compatibility_commands:
        run_command(command, description)
    
    # Then reinstall with compatible versions
    install_commands = [
        ("pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git", "Installing Unsloth from GitHub"),
        ("pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth-zoo.git", "Installing Unsloth Zoo from GitHub"),
        ("pip install bitsandbytes", "Installing BitsAndBytes"),
    ]
    
    success = True
    for command, description in install_commands:
        if not run_command(command, description):
            success = False
    
    return success


def install_unsloth():
    """Install Unsloth and related packages."""
    print("⚡ Installing Unsloth and related packages...")
    
    # First try the automatic installation approach
    print("🤖 Trying automatic Unsloth installation...")
    auto_install_cmd = "curl -s https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -"
    if run_command(auto_install_cmd, "Auto-installing Unsloth with optimal settings"):
        print("✅ Automatic installation successful!")
        return True
    
    print("⚠️ Auto-install failed, trying manual installation...")
    
    commands = [
        # Upgrade pip first
        ("pip install --upgrade pip", "Upgrading pip"),
        
        # Install specific transformers version (4.52.4 recommended)
        ("pip install transformers==4.52.4", "Installing compatible Transformers"),
        
        # Install core dependencies
        ("pip install datasets accelerate peft trl", "Installing core ML libraries"),
        
        # Install compatible PyTorch if needed
        ("pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", "Installing compatible PyTorch"),
        
        # Install Unsloth ecosystem - try simple approach first
        ("pip install unsloth", "Installing Unsloth (simple)"),
        
        # If that fails, try the GitHub approach
        ("pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"", "Installing Unsloth from GitHub"),
        
        # Install additional dependencies
        ("pip install bitsandbytes", "Installing BitsAndBytes"),
        ("pip install unsloth_zoo", "Installing Unsloth Zoo"),
        ("pip install timm", "Installing timm for Gemma 3N"),
        ("pip install comet-ml", "Installing Comet ML"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            if "Installing Unsloth" in description:
                print(f"⚠️ Warning: {description} failed")
                if "simple" not in description:
                    success = False
            else:
                print(f"⚠️ Warning: {description} failed, continuing...")
    
    return success


def install_optional_packages():
    """Install optional packages."""
    print("📦 Installing optional packages...")
    
    commands = [
        ("pip install comet-ml", "Installing Comet ML"),
    ]
    
    for command, description in commands:
        run_command(command, description)

def main():
    """Main setup function."""
    print("🚀 Setting up Gemma3N Training Pipeline Environment")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        print("❌ Python version check failed")
        return False
    
    # Check current installation
    print(f"\n📋 Checking current installation...")
    if not check_current_installation():
        print("🔧 Current installation has issues, attempting to fix...")
        if not fix_unsloth_installation():
            print("❌ Failed to fix Unsloth installation")
            
            # Try fresh installation
            print("🔄 Attempting fresh installation...")
            try:
                install_unsloth()
            except Exception as e:
                print(f"❌ Fresh installation failed: {e}")
                return False
    else:
        print("✅ Current installation looks good")
    
    # Install optional packages
    print(f"\n📋 Installing optional packages...")
    try:
        install_optional_packages()
    except Exception as e:
        print(f"⚠️ Optional packages installation failed: {e}")
    
    # Final check
    print(f"\n📋 Final environment check...")
    check_cuda()
    
    # Test Unsloth import
    print(f"\n🧪 Testing Unsloth import...")
    try:
        import unsloth
        from unsloth import FastLanguageModel
        print("✅ Unsloth import test passed")
        return True
    except Exception as e:
        print(f"❌ Unsloth import test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)