#!/usr/bin/env python3
"""
Test script for evaluation pipeline fixes.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evaluation_pipeline.config import ConfigManager
from src.evaluation_pipeline.managers.dataset_manager import EvaluationSample

def test_config_manager():
    """Test ConfigManager functionality."""
    print("Testing ConfigManager...")
    
    # Test loading config from YAML
    config_path = "configs/evaluation_quick.yaml"
    config = ConfigManager.from_yaml(config_path)
    
    print(f"✓ Config loaded successfully")
    print(f"  - Model: {config.get_model_config().name}")
    print(f"  - Dataset: {config.get_dataset_config().dataset_id}")
    print(f"  - Experiment: {config.get_experiment_config().experiment_name}")
    
    return config

def test_dataset_manager(config):
    """Test DatasetManager functionality."""
    print("\nTesting DatasetManager...")
    
    from src.evaluation_pipeline.managers.dataset_manager import DatasetManager
    
    dataset_manager = DatasetManager(config)
    samples = dataset_manager.load_dataset()
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Samples: {len(samples)}")
    print(f"  - First sample question: {samples[0].question[:100]}...")
    
    return samples

def test_inference_engine_padding():
    """Test the padding fix in InferenceEngine."""
    print("\nTesting InferenceEngine padding fix...")
    
    import torch
    from src.evaluation_pipeline.inference.inference_engine import InferenceEngine
    
    # Create mock tokenizer with eos_token_id
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 2
        
        def apply_chat_template(self, messages, **kwargs):
            # Mock implementation
            return {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
    
    # Create mock model and config
    class MockModel:
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    
    class MockConfig:
        def get_generation_config(self):
            return {"max_new_tokens": 10}
    
    # Test the padding logic
    tokenizer = MockTokenizer()
    model = MockModel()
    config = MockConfig()
    
    engine = InferenceEngine(model, tokenizer, config)
    
    # Create batch inputs with different lengths
    batch_inputs = [
        {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        },
        {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
    ]
    
    try:
        padded_inputs = engine._prepare_batch_inputs(batch_inputs)
        print(f"✓ Padding fix works correctly")
        print(f"  - Input shape: {padded_inputs['input_ids'].shape}")
        print(f"  - Attention mask shape: {padded_inputs['attention_mask'].shape}")
    except Exception as e:
        print(f"✗ Padding fix failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Evaluation Pipeline Fixes\n")
    
    try:
        # Test 1: ConfigManager
        config = test_config_manager()
        
        # Test 2: DatasetManager
        samples = test_dataset_manager(config)
        
        # Test 3: InferenceEngine padding
        padding_works = test_inference_engine_padding()
        
        print("\n🎉 All tests completed!")
        
        if padding_works:
            print("✅ Padding fix is working correctly")
        else:
            print("❌ Padding fix needs more work")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
