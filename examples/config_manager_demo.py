"""
ConfigManager Demo Script
========================

This script demonstrates how to use the new ConfigManager system for unified
configuration management across different YAML formats.

Usage:
    python examples/config_manager_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_pipeline.config.config_manager import create_config_manager, ConfigManager


def demo_unified_config():
    """Demo using the new unified config format."""
    print("=" * 60)
    print("🔧 ConfigManager Demo - Unified Format")
    print("=" * 60)
    
    # Load unified config
    config_manager = create_config_manager(
        config_path="configs/unified_training_config.yaml",
        apply_env=True
    )
    
    print("📊 Configuration loaded successfully!")
    print(config_manager.summary())
    
    # Access specific config sections
    print("\n🤖 Model Config:")
    print(f"   Name: {config_manager.model.name}")
    print(f"   Max length: {config_manager.model.max_seq_length}")
    print(f"   4-bit quantization: {config_manager.model.load_in_4bit}")
    
    print("\n📚 Dataset Config:")
    print(f"   Name: {config_manager.dataset.name}")
    print(f"   Train split: {config_manager.dataset.train_split}")
    print(f"   Text field: {config_manager.dataset.text_field}")
    
    print("\n🎯 Training Config:")
    print(f"   Max steps: {config_manager.training.max_steps}")
    print(f"   Learning rate: {config_manager.training.learning_rate:.2e}")
    print(f"   Batch size: {config_manager.training.per_device_train_batch_size}")
    print(f"   Effective batch size: {config_manager.get_effective_batch_size()}")
    
    print("\n🔧 LoRA Config:")
    print(f"   Rank: {config_manager.lora.r}")
    print(f"   Alpha: {config_manager.lora.alpha}")
    print(f"   Target modules: {config_manager.lora.target_modules}")
    
    print("\n📁 Output Config:")
    print(f"   Output directory: {config_manager.output.get_output_dir()}")
    print(f"   Save formats: {config_manager.output.save_formats}")
    
    return config_manager


def demo_legacy_configs():
    """Demo loading legacy flat config formats."""
    print("\n" + "=" * 60)
    print("🔄 ConfigManager Demo - Legacy Format Compatibility")
    print("=" * 60)
    
    legacy_configs = [
        "configs/development.yaml",
        "configs/production.yaml", 
        "configs/training_config.yaml"
    ]
    
    for config_path in legacy_configs:
        try:
            print(f"\n📄 Loading: {config_path}")
            config_manager = create_config_manager(
                config_path=config_path,
                apply_env=False  # Skip env for demo
            )
            
            print(f"   ✅ Loaded successfully!")
            print(f"   🤖 Model: {config_manager.model.name}")
            print(f"   📚 Dataset: {config_manager.dataset.name}")
            print(f"   🎯 Max steps: {config_manager.training.max_steps}")
            print(f"   🔧 LoRA rank: {config_manager.lora.r}")
            print(f"   📁 Output: {config_manager.output.get_output_dir()}")
            
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")


def demo_overrides():
    """Demo CLI and environment overrides."""
    print("\n" + "=" * 60)
    print("⚙️ ConfigManager Demo - Overrides")
    print("=" * 60)
    
    # Demo CLI overrides
    print("\n🖥️ CLI Overrides Demo:")
    cli_overrides = {
        'experiment_name': 'demo-experiment',
        'max_steps': 50,
        'learning_rate': 1e-4,
        'lora_r': 8,
        'output_dir': 'outputs/demo'
    }
    
    config_manager = create_config_manager(
        config_path="configs/unified_training_config.yaml",
        apply_env=False,
        cli_overrides=cli_overrides
    )
    
    print(f"   ✅ Applied CLI overrides:")
    print(f"   📛 Experiment name: {config_manager.output.experiment_name}")
    print(f"   🎯 Max steps: {config_manager.training.max_steps}")
    print(f"   📈 Learning rate: {config_manager.training.learning_rate:.2e}")
    print(f"   🔧 LoRA rank: {config_manager.lora.r}")
    print(f"   📁 Output: {config_manager.output.get_output_dir()}")


def demo_manager_injection():
    """Demo how managers receive specific config sections."""
    print("\n" + "=" * 60)
    print("🏭 ConfigManager Demo - Manager Dependency Injection")
    print("=" * 60)
    
    config_manager = create_config_manager(
        config_path="configs/unified_training_config.yaml"
    )
    
    # Show how managers are initialized with specific config sections
    print("\n🧪 ExperimentManager receives:")
    print(f"   📁 Output config: {config_manager.output.experiment_name}")
    print(f"   📊 Comet config: enabled={config_manager.comet.enabled}")
    
    print("\n💾 CheckpointManager receives:")
    print(f"   📁 Output config: {config_manager.output.save_formats}")
    print(f"   🌐 Hub config: push_to_hub={config_manager.raw_config.get('hub', {}).get('push_to_hub', False)}")
    
    print("\n🏭 ModelFactory receives:")
    print(f"   🤖 Model config: {config_manager.model.name}")
    print(f"   🔧 LoRA config: rank={config_manager.lora.r}")
    
    print("\n📊 DatasetFactory receives:")
    print(f"   📚 Dataset config: {config_manager.dataset.name}")
    print(f"   🎯 Training config: batch_size={config_manager.training.per_device_train_batch_size}")


def demo_validation():
    """Demo configuration validation."""
    print("\n" + "=" * 60)
    print("✅ ConfigManager Demo - Validation")
    print("=" * 60)
    
    # Demo validation with invalid config
    print("\n❌ Testing invalid configuration:")
    
    invalid_config = {
        'model': {'name': '', 'max_seq_length': -1},  # Invalid: empty name, negative length
        'training': {'max_steps': 0, 'learning_rate': -1},  # Invalid: zero steps, negative LR
        'lora': {'r': 0, 'alpha': -1},  # Invalid: zero rank, negative alpha
        'output': {'base_dir': '', 'experiment_name': ''}  # Invalid: empty values
    }
    
    try:
        config_manager = create_config_manager(config_data=invalid_config)
        print("   ❌ Validation should have failed!")
    except Exception as e:
        print(f"   ✅ Validation correctly failed: {e}")
    
    # Demo validation with valid config
    print("\n✅ Testing valid configuration:")
    try:
        config_manager = create_config_manager(
            config_path="configs/unified_training_config.yaml"
        )
        print("   ✅ Validation passed successfully!")
    except Exception as e:
        print(f"   ❌ Unexpected validation error: {e}")


def main():
    """Run all ConfigManager demos."""
    print("🚀 Starting ConfigManager Demo...")
    
    try:
        # Demo 1: Unified config format
        demo_unified_config()
        
        # Demo 2: Legacy config compatibility  
        demo_legacy_configs()
        
        # Demo 3: Overrides
        demo_overrides()
        
        # Demo 4: Manager dependency injection
        demo_manager_injection()
        
        # Demo 5: Validation
        demo_validation()
        
        print("\n" + "=" * 60)
        print("🎉 All ConfigManager demos completed successfully!")
        print("=" * 60)
        
        print("\n💡 Key Benefits of the new ConfigManager system:")
        print("   ✅ Unified config format across all YAML files")
        print("   ✅ Type-safe access to configuration sections")
        print("   ✅ Proper dependency injection for managers")
        print("   ✅ Environment variable and CLI override support")
        print("   ✅ Comprehensive validation with clear error messages")
        print("   ✅ Backward compatibility with legacy formats")
        print("   ✅ Better debugging and error tracing")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
