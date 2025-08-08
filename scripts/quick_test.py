#!/usr/bin/env python3
"""Quick test script for the training pipeline."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_pipeline.cli.train_gemma import main as train_main


def quick_test():
    """Run a quick test of the training pipeline."""
    
    print("🧪 Running quick test of Gemma3N training pipeline...")
    print("=" * 60)
    
    # Override command line arguments for quick test
    original_argv = sys.argv.copy()
    
    sys.argv = [
        "train_gemma.py",
        "--quick-test",
        "--max-steps", "5",
        "--batch-size", "1", 
        "--gradient-accumulation-steps", "2",
        "--logging-steps", "1",
        "--save-steps", "3",
        "--disable-comet",
        "--log-level", "INFO",
        "--test-model",
        "--save-formats", "lora"
    ]
    
    try:
        # Run training
        train_main()
        
        print("\n" + "=" * 60)
        print("✅ Quick test completed successfully!")
        print("🎉 The training pipeline is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        print("🔧 Please check your setup and try again.")
        raise
    
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    quick_test()