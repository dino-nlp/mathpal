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


def install_unsloth():
    """Install Unsloth and related packages."""
    print("⚡ Installing Unsloth and related packages...")
    
    commands = [
        ("pip install unsloth", "Installing Unsloth"),
        ("pip install --no-deps --upgrade timm", "Installing timm only for Gemma 3N"),
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
    
    # Install packages
    steps = [
        (install_unsloth, "Installing Unsloth ecosystem"),
        (install_optional_packages, "Installing optional packages"),
    ]
    
    for step_func, step_name in steps:
        print(f"\n📋 {step_name}...")
        try:
            step_func()
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            return False
        
    check_cuda()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)