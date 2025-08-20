#!/usr/bin/env python3
"""
Script to test model loading safely.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_loading():
    """Test model loading with safe configuration"""
    print("🔍 Testing Model Loading")
    print("=" * 40)
    
    try:
        # Import required modules
        from unsloth import FastModel
        from inference_pipeline.config import settings
        
        print(f"📊 Model ID: {settings.MODEL_ID}")
        print(f"📊 Max Input Tokens: {settings.MAX_INPUT_TOKENS}")
        
        # Test model loading
        print("🔄 Loading model...")
        model, processor = FastModel.from_pretrained(
            model_name=settings.MODEL_ID,
            dtype=None,  # Auto-detect
            max_seq_length=settings.MAX_INPUT_TOKENS,
            load_in_4bit=True,
            load_in_8bit=False,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print("✅ Model loaded successfully!")
        
        # Test processor
        print("🔄 Testing processor...")
        from unsloth import get_chat_template
        processor = get_chat_template(processor, "gemma-3n")
        print("✅ Processor configured successfully!")
        
        # Test inference preparation
        print("🔄 Preparing for inference...")
        FastModel.for_inference(model)
        model.eval()
        print("✅ Model prepared for inference!")
        
        # Test basic generation
        print("🔄 Testing basic generation...")
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Hello, how are you?",
            }]
        }]
        
        input_ids = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True, 
            return_dict=True,
            return_tensors="pt",
        )
        
        # Determine device
        if hasattr(model, 'device'):
            device = model.device
        elif hasattr(model, 'hf_device_map'):
            device_map = model.hf_device_map
            if device_map:
                device = list(device_map.values())[0]
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Move input_ids to device
        input_ids = {k: v.to(device) if hasattr(v, 'to') else v for k, v in input_ids.items()}
        
        with torch.no_grad():
            response = model.generate(
                **input_ids,
                max_new_tokens=10,  # Short for testing
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                pad_token_id=processor.eos_token_id,
                eos_token_id=processor.eos_token_id,
            )
        
        answer = processor.batch_decode(response, skip_special_tokens=False)[0]
        print(f"✅ Generation test successful!")
        print(f"📝 Response: {answer[:100]}...")
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n🎉 All tests passed! Model is ready for evaluation.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run model loading test"""
    print("🔍 MathPal Model Loading Test")
    print("=" * 50)
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCH_LOGS"] = "off"
    
    # Disable TorchDynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    
    success = test_model_loading()
    
    if success:
        print("\n✅ Model loading test completed successfully!")
        print("🎯 You can now run: make evaluate-llm-compatible")
    else:
        print("\n❌ Model loading test failed!")
        print("💡 Check the error messages above for details")

if __name__ == "__main__":
    main()
