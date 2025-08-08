#!/usr/bin/env python3
"""
Debug script để kiểm tra SFTTrainer API và tìm cách sử dụng đúng
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_sfttrainer_signature():
    """Test SFTTrainer signature để xem có nhận tokenizer không"""
    print("🔍 Testing SFTTrainer API signature...")
    
    try:
        from trl import SFTTrainer
        import inspect
        
        # Get SFTTrainer __init__ signature
        signature = inspect.signature(SFTTrainer.__init__)
        print(f"SFTTrainer.__init__ parameters:")
        
        for param_name, param in signature.parameters.items():
            if param_name != 'self':
                print(f"  - {param_name}: {param}")
        
        # Check specifically for tokenizer
        has_tokenizer = 'tokenizer' in signature.parameters
        print(f"\n✅ Has 'tokenizer' parameter: {has_tokenizer}")
        
        return has_tokenizer
        
    except Exception as e:
        print(f"❌ Error inspecting SFTTrainer: {e}")
        return False

def test_alternative_approaches():
    """Test alternative approaches để tạo SFTTrainer"""
    print("\n🧪 Testing Alternative Approaches...")
    
    try:
        from trl import SFTTrainer, SFTConfig
        
        # Approach 1: Minimal parameters
        print("Approach 1: Minimal parameters only...")
        try:
            config = SFTConfig(
                dataset_text_field="text",
                output_dir="./test",
                max_steps=1,
            )
            print("  ✅ SFTConfig created successfully")
            
            # Don't actually create trainer without model
            print("  ✅ Would work: SFTTrainer(model=model, args=config)")
            
        except Exception as e:
            print(f"  ❌ Error with minimal approach: {e}")
        
        # Approach 2: Check if we need to use dataset parameter instead
        print("\nApproach 2: Using train_dataset parameter...")
        try:
            config = SFTConfig(
                dataset_text_field="text", 
                output_dir="./test",
                max_steps=1,
            )
            print("  ✅ Could use: SFTTrainer(model=model, train_dataset=dataset, args=config)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing alternatives: {e}")
        return False

def test_import_versions():
    """Test các versions của libraries"""
    print("\n📦 Checking Library Versions...")
    
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except:
        print("  transformers: Not available")
    
    try:
        import trl
        print(f"  trl: {trl.__version__}")
    except:
        print("  trl: Not available")
    
    try:
        import unsloth
        print(f"  unsloth: Available (version not easily accessible)")
    except:
        print("  unsloth: Not available")
    
    try:
        import torch
        print(f"  torch: {torch.__version__}")
    except:
        print("  torch: Not available")

def test_working_pattern():
    """Test exact pattern từ working notebook"""
    print("\n📝 Testing Working Notebook Pattern...")
    
    print("Working notebook uses:")
    print("```python")
    print("trainer = SFTTrainer(")
    print("    model=model,")
    print("    tokenizer=processor,  # ← This parameter") 
    print("    train_dataset=train_dataset,")
    print("    args=training_args,")
    print(")")
    print("```")
    
    print("\nOur current code uses:")
    print("```python") 
    print("trainer = SFTTrainer(")
    print("    model=self.model_manager.model,")
    print("    tokenizer=self.model_manager.tokenizer,  # ← This parameter")
    print("    train_dataset=datasets['train'],")
    print("    args=training_args,")
    print(")")
    print("```")
    
    print("\n🤔 Possible issues:")
    print("1. Version difference between working notebook and our environment")
    print("2. Different import order causing API changes")
    print("3. Need to use 'processing_class' instead of 'tokenizer'")
    print("4. Need to remove tokenizer parameter entirely")

if __name__ == "__main__":
    print("🔧 Debugging SFTTrainer API Issues")
    print("=" * 50)
    
    test_import_versions()
    has_tokenizer = test_sfttrainer_signature() 
    test_alternative_approaches()
    test_working_pattern()
    
    print("\n" + "=" * 50)
    print("💡 RECOMMENDATIONS:")
    
    if has_tokenizer:
        print("✅ SFTTrainer HAS tokenizer parameter")
        print("   → Try different import order or check model_manager.tokenizer")
    else:
        print("❌ SFTTrainer DOES NOT have tokenizer parameter")  
        print("   → Try removing tokenizer parameter entirely")
        print("   → Or use 'processing_class' parameter")
        print("   → Or set tokenizer in SFTConfig")
    
    print("\nNext steps:")
    print("1. Try removing tokenizer parameter")
    print("2. Try using processing_class parameter")  
    print("3. Check if tokenizer should be set in model preparation")
    print("4. Match exact transformers/trl versions from working notebook")