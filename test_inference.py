#!/usr/bin/env python3
"""
Test script for InferenceEngine after fixing tokenization issues.
"""

import sys
import os
sys.path.append('/content')  # Add to path for Colab

from src.training_pipeline.inference.inference_engine import InferenceEngine
import torch
from unsloth import FastLanguageModel

def setup_model_and_tokenizer():
    """Setup model and tokenizer for testing."""
    print("🔧 Setting up model and tokenizer...")
    
    # Load model and tokenizer (adjust model name as needed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",  # Change to your model
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
    
    return model, tokenizer

def test_basic_inference(inference_engine):
    """Test basic inference functionality."""
    print("\n" + "="*50)
    print("🧪 TESTING BASIC INFERENCE")
    print("="*50)
    
    test_questions = [
        "Tính 15 + 27 = ?",
        "Tìm chu vi hình vuông cạnh 5cm?",
        "Giải phương trình: x + 10 = 25"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i} ---")
        print(f"Q: {question}")
        
        try:
            response = inference_engine.generate(question)
            print(f"A: {response}")
            print("✅ Success")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_different_configs(inference_engine):
    """Test different generation configs."""
    print("\n" + "="*50)
    print("🎛️ TESTING DIFFERENT CONFIGS")
    print("="*50)
    
    configs = InferenceEngine.get_recommended_configs()
    test_question = "Tính diện tích hình chữ nhật có chiều dài 8m, chiều rộng 5m?"
    
    print(f"Test question: {test_question}")
    
    for config_name, config in configs.items():
        print(f"\n--- {config_name.upper()} CONFIG ---")
        print(f"Config: {config}")
        
        try:
            response = inference_engine.generate(
                question=test_question,
                generation_config=config
            )
            print(f"Response: {response}")
            print("✅ Success")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_benchmark_function(inference_engine):
    """Test the fixed benchmark function."""
    print("\n" + "="*50)
    print("⏱️ TESTING BENCHMARK FUNCTION")
    print("="*50)
    
    benchmark_questions = [
        "Tính 25 + 15 = ?",
        "Tìm chu vi hình tròn bán kính 3cm?",
        "Giải phương trình: 2x + 4 = 14"
    ]
    
    try:
        print(f"Running benchmark with {len(benchmark_questions)} questions...")
        results = inference_engine.benchmark_inference(
            questions=benchmark_questions,
            num_runs=2
        )
        
        print("\n📊 BENCHMARK RESULTS:")
        print(f"- Average time: {results['avg_time']:.2f}s")
        print(f"- Average tokens: {results['avg_tokens']:.1f}")
        print(f"- Average tokens/second: {results['avg_tokens_per_second']:.1f}")
        print("✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

def test_batch_generation(inference_engine):
    """Test batch generation."""
    print("\n" + "="*50)
    print("📦 TESTING BATCH GENERATION")
    print("="*50)
    
    batch_questions = [
        "Tính 12 + 8 = ?",
        "Tìm diện tích hình vuông cạnh 4cm?",
        "Trong lớp có 30 học sinh, 18 học sinh nam. Có bao nhiêu học sinh nữ?"
    ]
    
    try:
        print(f"Generating responses for {len(batch_questions)} questions...")
        responses = inference_engine.generate_batch(
            questions=batch_questions,
            batch_size=2
        )
        
        print("\n📝 BATCH RESULTS:")
        for i, (q, a) in enumerate(zip(batch_questions, responses), 1):
            print(f"\n{i}. Q: {q}")
            print(f"   A: {a}")
        
        print("✅ Batch generation completed!")
        
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")

def test_edge_cases(inference_engine):
    """Test edge cases and error handling."""
    print("\n" + "="*50)
    print("🚨 TESTING EDGE CASES")
    print("="*50)
    
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "A" * 1000,  # Very long input
        "Câu hỏi này có dấu tiếng Việt với ký tự đặc biệt: !@#$%^&*()",  # Special characters
    ]
    
    for i, test_input in enumerate(edge_cases, 1):
        print(f"\n--- Edge Case {i} ---")
        print(f"Input: '{test_input[:50]}{'...' if len(test_input) > 50 else ''}'")
        
        try:
            response = inference_engine.generate(test_input)
            print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            print("✅ Handled successfully")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main test function."""
    print("🚀 INFERENCE ENGINE TEST SUITE")
    print("="*50)
    
    try:
        # Setup
        model, tokenizer = setup_model_and_tokenizer()
        
        # Create inference engine
        inference_engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.9
        )
        
        # Run tests
        test_basic_inference(inference_engine)
        test_different_configs(inference_engine)
        test_batch_generation(inference_engine)
        test_edge_cases(inference_engine)
        
        # Test the fixed benchmark function last
        test_benchmark_function(inference_engine)
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS COMPLETED!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ SETUP ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
