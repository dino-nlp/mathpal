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
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Success: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description or command}")
        print(f"Error: {e.stderr}")
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


def install_torch():
    """Install PyTorch with CUDA support."""
    print("🔥 Installing PyTorch with CUDA support...")
    
    command = (
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    
    return run_command(command, "Installing PyTorch")


def install_unsloth():
    """Install Unsloth and related packages."""
    print("⚡ Installing Unsloth and related packages...")
    
    commands = [
        ("pip install --no-deps xformers==0.0.29.post3", "Installing xformers"),
        ("pip install --no-deps bitsandbytes accelerate", "Installing bitsandbytes and accelerate"),
        ("pip install --no-deps peft trl", "Installing PEFT and TRL"),
        ("pip install --no-deps unsloth", "Installing Unsloth"),
        ("pip install transformers datasets tokenizers", "Installing transformers ecosystem"),
        ("pip install sentencepiece protobuf", "Installing tokenization dependencies")
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success


def install_optional_packages():
    """Install optional packages."""
    print("📦 Installing optional packages...")
    
    commands = [
        ("pip install comet-ml", "Installing Comet ML (optional)"),
        ("pip install wandb", "Installing Weights & Biases (optional)"),
        ("pip install pyyaml", "Installing YAML support"),
        ("pip install rich", "Installing Rich for better output"),
        ("pip install click", "Installing Click for CLI")
    ]
    
    for command, description in commands:
        run_command(command, description)


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "outputs",
        "logs", 
        "configs",
        "examples/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def setup_environment_file():
    """Create example environment file."""
    print("🔧 Creating example environment file...")
    
    env_content = """# Comet ML Configuration (optional)
# Get your API key from https://www.comet.ml/
COMET_API_KEY=your-api-key-here
COMET_WORKSPACE=your-workspace
COMET_PROJECT=gemma3n-finetuning

# HuggingFace Hub Configuration (optional)
# Get your token from https://huggingface.co/settings/tokens
HF_TOKEN=your-token-here

# Other optional settings
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0
"""
    
    env_file = Path(".env.example")
    env_file.write_text(env_content)
    print(f"✅ Created {env_file}")
    print("   Copy this to .env and fill in your credentials")


def verify_installation():
    """Verify the installation."""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import peft
        print(f"✅ PEFT {peft.__version__}")
        
        import trl
        print(f"✅ TRL {trl.__version__}")
        
        try:
            import unsloth
            print("✅ Unsloth installed")
        except ImportError:
            print("⚠️ Unsloth not found - may need manual installation")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} device(s)")
        else:
            print("⚠️ CUDA not available")
        
        print("✅ Installation verification completed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Gemma3N Training Pipeline Environment")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        print("❌ Python version check failed")
        return False
    
    # Install packages
    steps = [
        (install_torch, "Installing PyTorch"),
        (install_unsloth, "Installing Unsloth ecosystem"),
        (install_optional_packages, "Installing optional packages"),
        (create_directories, "Creating directories"),
        (setup_environment_file, "Setting up environment"),
    ]
    
    for step_func, step_name in steps:
        print(f"\n📋 {step_name}...")
        try:
            step_func()
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            return False
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    if verify_installation():
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Copy .env.example to .env and fill in your credentials")
        print("2. Run a quick test: python scripts/quick_test.py")
        print("3. Start training: python -m training_pipeline.cli.train_gemma --quick-test")
        print("\n📚 Check README.md for more information")
        return True
    else:
        print("\n❌ Setup verification failed")
        print("Please check the error messages above and try again")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)