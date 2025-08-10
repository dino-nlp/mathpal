#!/usr/bin/env python3
"""
Quick test script to verify the fixed benchmark_inference function.
"""

import sys
sys.path.append('/content')  # For Colab

def quick_benchmark_test(model, tokenizer):
    """Quick test of the fixed benchmark function."""
    from src.training_pipeline.inference.inference_engine import InferenceEngine
    
    print("🚀 Quick Benchmark Test")
    print("="*40)
    
    # Create inference engine
    inference_engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=32,
        temperature=0.7
    )
    
    # Test questions
    test_questions = [
        "Tính 10 + 15 = ?",
        "Tìm chu vi hình vuông cạnh 6cm?",
        "Giải: x + 8 = 20"
    ]
    
    print(f"Testing with {len(test_questions)} questions...")
    print("Questions:")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")
    
    try:
        # Run benchmark
        results = inference_engine.benchmark_inference(
            questions=test_questions,
            num_runs=2
        )
        
        print("\n✅ SUCCESS! Benchmark completed without errors.")
        print(f"\n📊 Results:")
        print(f"   • Average time: {results['avg_time']:.2f}s")
        print(f"   • Average tokens: {results['avg_tokens']:.1f}")
        print(f"   • Speed: {results['avg_tokens_per_second']:.1f} tokens/sec")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Please run this with your model and tokenizer:")
    print("quick_benchmark_test(your_model, your_tokenizer)")
