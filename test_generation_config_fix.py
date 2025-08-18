#!/usr/bin/env python3
"""
Test script for GenerationConfig fix.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_generation_config_conversion():
    """Test GenerationConfig to dict conversion."""
    print("Testing GenerationConfig conversion...")
    
    from src.evaluation_pipeline.config import ConfigManager
    
    # Load config
    config = ConfigManager.from_yaml("configs/evaluation_quick.yaml")
    gen_config = config.get_generation_config()
    
    print(f"✓ GenerationConfig loaded: {type(gen_config)}")
    print(f"  - Content: {gen_config}")
    
    # Test conversion
    if hasattr(gen_config, 'model_dump'):
        gen_dict = gen_config.model_dump()
        print(f"✓ Converted using model_dump(): {type(gen_dict)}")
    elif hasattr(gen_config, 'dict'):
        gen_dict = gen_config.dict()
        print(f"✓ Converted using dict(): {type(gen_dict)}")
    else:
        print(f"✗ No conversion method found")
        return False
    
    print(f"  - Dict content: {gen_dict}")
    
    # Test that it's a proper dict
    if isinstance(gen_dict, dict):
        print(f"✓ Successfully converted to dict")
        return True
    else:
        print(f"✗ Conversion failed - not a dict")
        return False

def main():
    """Run the test."""
    print("🧪 Testing GenerationConfig Fix\n")
    
    try:
        success = test_generation_config_conversion()
        
        if success:
            print("\n🎉 Test passed!")
            print("✅ GenerationConfig conversion works correctly")
        else:
            print("\n❌ Test failed!")
            print("❌ GenerationConfig conversion needs more work")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
