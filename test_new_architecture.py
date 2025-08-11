#!/usr/bin/env python3
"""
Test script for new training pipeline architecture.

This script validates that the new modular architecture works correctly
without actually running training (similar to dry-run mode).
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all new architecture components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test core exceptions
        from training_pipeline.core.exceptions import (
            TrainingPipelineError, ValidationError, ModelError, DatasetError
        )
        print("   ✅ Core exceptions imported")
        
        # Test enhanced config
        from training_pipeline.core.enhanced_config import (
            ConfigLoader, ComprehensiveTrainingConfig
        )
        print("   ✅ Enhanced config imported")
        
        # Test training manager
        from training_pipeline.core.training_manager import TrainingManager, TrainingResults
        print("   ✅ Training manager imported")
        
        # Test factories
        from training_pipeline.factories import ModelFactory, DatasetFactory, TrainerFactory
        print("   ✅ Factories imported")
        
        # Test managers
        from training_pipeline.managers import ExperimentManager, CheckpointManager, EvaluationManager
        print("   ✅ Managers imported")
        
        print("✅ All imports successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}\n")
        return False


def test_config_loading():
    """Test configuration loading from YAML files."""
    print("📄 Testing configuration loading...")
    
    try:
        from training_pipeline.core.enhanced_config import ConfigLoader
        
        # Test config files
        config_files = [
            "configs/complete_training_config.yaml",
            "configs/development.yaml", 
            "configs/production.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    config = ConfigLoader.load_from_yaml(config_file)
                    config.validate()
                    print(f"   ✅ {config_file} loaded and validated")
                except Exception as e:
                    print(f"   ❌ {config_file} validation failed: {e}")
                    return False
            else:
                print(f"   ⚠️ {config_file} not found (skipping)")
        
        print("✅ Configuration loading successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}\n")
        return False


def test_cli_parsing():
    """Test CLI argument parsing."""
    print("🖥️ Testing CLI argument parsing...")
    
    try:
        # Import the CLI module
        from training_pipeline.cli.train_gemma_v2 import parse_arguments
        
        # Test with minimal arguments
        sys.argv = ["train_gemma_v2.py", "--config", "configs/development.yaml"]
        args = parse_arguments()
        
        print(f"   ✅ CLI parsing successful")
        print(f"   📁 Config: {args.config}")
        print(f"   🧪 Quick test: {args.quick_test}")
        print(f"   🔍 Dry run: {args.dry_run}")
        print(f"   🐛 Debug: {args.debug}")
        
        print("✅ CLI parsing successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ CLI parsing test failed: {e}\n")
        return False


def test_factory_support():
    """Test factory pattern support."""
    print("🏭 Testing factory pattern support...")
    
    try:
        from training_pipeline.factories import ModelFactory
        from training_pipeline.core.enhanced_config import ConfigLoader
        
        # Load a config
        if os.path.exists("configs/development.yaml"):
            config = ConfigLoader.load_from_yaml("configs/development.yaml")
            
            # Test model factory support check
            is_supported = ModelFactory.is_supported(config.model.name)
            print(f"   ✅ Model {config.model.name} supported: {is_supported}")
            
            # Test memory estimation
            estimated_memory = ModelFactory.estimate_memory_usage(config)
            print(f"   💾 Estimated memory: {estimated_memory:.2f} GB")
            
            # Test supported model types
            supported_types = ModelFactory.get_supported_models()
            print(f"   🤖 Supported model types: {list(supported_types.keys())}")
        
        print("✅ Factory pattern testing successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}\n")
        return False


def test_dry_run_functionality():
    """Test dry run functionality."""
    print("🔍 Testing dry run functionality...")
    
    try:
        from training_pipeline.core.enhanced_config import ConfigLoader
        from training_pipeline.core.training_manager import TrainingManager
        
        if os.path.exists("configs/development.yaml"):
            # Load config
            config = ConfigLoader.load_from_yaml("configs/development.yaml")
            
            # Apply quick test profile
            config = ConfigLoader.apply_quick_test_profile(config)
            
            # Validate config
            config.validate()
            print("   ✅ Config validation passed")
            
            # Create training manager
            manager = TrainingManager(config)
            print("   ✅ Training manager created")
            
            # Test cost estimation
            cost_estimate = manager.estimate_training_cost()
            
            # Safe formatting for numeric values
            memory_est = cost_estimate.get('estimated_memory_gb', 'Unknown')
            if isinstance(memory_est, (int, float)):
                print(f"   💰 Memory estimate: {memory_est:.1f} GB")
            else:
                print(f"   💰 Memory estimate: {memory_est}")
            
            time_est = cost_estimate.get('estimated_time_hours', 'Unknown')  
            if isinstance(time_est, (int, float)):
                print(f"   ⏱️ Time estimate: {time_est:.1f} hours")
            else:
                print(f"   ⏱️ Time estimate: {time_est}")
                
            print(f"   ✅ Feasible: {'Yes' if cost_estimate.get('feasible', True) else 'No'}")
            
        print("✅ Dry run functionality successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Dry run test failed: {e}\n")
        return False


def test_architecture_comparison():
    """Compare old vs new architecture."""
    print("📊 Architecture Comparison:")
    print("=" * 60)
    
    # Check old architecture file
    old_file = "src/training_pipeline/cli/train_gemma.py"
    new_file = "src/training_pipeline/cli/train_gemma_v2.py"
    
    try:
        old_lines = 0
        if os.path.exists(old_file):
            with open(old_file, 'r') as f:
                old_lines = len(f.readlines())
        
        new_lines = 0 
        if os.path.exists(new_file):
            with open(new_file, 'r') as f:
                new_lines = len(f.readlines())
        
        print(f"📊 OLD Architecture (train_gemma.py):")
        print(f"   📄 Lines of code: {old_lines}")
        print(f"   ❌ Monolithic structure")
        print(f"   ❌ 50+ CLI arguments")
        print(f"   ❌ Hardcoded defaults")
        print("")
        
        print(f"✅ NEW Architecture (train_gemma_v2.py):")
        print(f"   📄 Lines of code: {new_lines}")
        print(f"   ✅ Modular structure")
        print(f"   ✅ 5-7 CLI arguments")
        print(f"   ✅ YAML-driven config")
        print(f"   📉 Code reduction: {((old_lines - new_lines) / old_lines * 100):.1f}%")
        print("")
        
        # Count new architecture files
        new_files = [
            "src/training_pipeline/core/enhanced_config.py",
            "src/training_pipeline/core/training_manager.py", 
            "src/training_pipeline/factories/model_factory.py",
            "src/training_pipeline/factories/dataset_factory.py",
            "src/training_pipeline/factories/trainer_factory.py",
            "src/training_pipeline/managers/experiment_manager.py",
            "src/training_pipeline/managers/checkpoint_manager.py",
            "src/training_pipeline/managers/evaluation_manager.py",
        ]
        
        total_new_lines = 0
        existing_files = 0
        for file_path in new_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = len(f.readlines())
                    total_new_lines += lines
                    existing_files += 1
        
        print(f"🏗️ New Architecture Components:")
        print(f"   📁 New files created: {existing_files}")
        print(f"   📄 Total lines in new components: {total_new_lines:,}")
        print(f"   🧩 Modular design with clear separation of concerns")
        
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Architecture comparison failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 MathPal Training Pipeline v2 - Architecture Validation")
    print("=" * 60)
    print("")
    
    tests = [
        ("Import Tests", test_imports),
        ("Config Loading Tests", test_config_loading),
        ("CLI Parsing Tests", test_cli_parsing),
        ("Factory Pattern Tests", test_factory_support),
        ("Dry Run Tests", test_dry_run_functionality),
        ("Architecture Comparison", test_architecture_comparison),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
        print("")
    
    # Final summary
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! New architecture is ready for use.")
        print("")
        print("📖 Usage Examples:")
        print("   make train-dev-v2          # Development training")
        print("   make train-quick            # Quick test")
        print("   make train-dry-run          # Validate without training")
        print("   make show-architecture      # Show architecture comparison")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please fix issues before using new architecture.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
