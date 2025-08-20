#!/usr/bin/env python3
"""
Script to test evaluation safely.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_evaluation():
    """Test evaluation with minimal setup"""
    print("🔍 Testing Evaluation Setup")
    print("=" * 40)
    
    try:
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["TORCH_LOGS"] = "off"
        
        # Disable TorchDynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
        
        # Import required modules
        from inference_pipeline.evaluation.evaluate import get_scoring_metrics
        from inference_pipeline.evaluation.progress_metrics import ProgressStyle, ProgressLevenshteinRatio
        
        print("✅ Imports successful")
        
        # Test metrics creation
        print("🔄 Testing metrics creation...")
        metrics = get_scoring_metrics(use_progress_metrics=True, track_progress=False)
        print(f"✅ Created {len(metrics)} metrics")
        
        for i, metric in enumerate(metrics):
            print(f"   {i+1}. {metric.name}")
        
        # Test individual metric
        print("🔄 Testing ProgressStyle metric...")
        style_metric = ProgressStyle(track_progress=False)
        
        # Test with sample data
        test_output = "The answer is 42 because 6 * 7 = 42"
        test_problem = "What is 6 times 7?"
        
        result = style_metric.score(output=test_output, problem=test_problem)
        print(f"✅ Style metric test successful: {result.value:.3f}")
        
        # Test Levenshtein metric
        print("🔄 Testing ProgressLevenshteinRatio metric...")
        levenshtein_metric = ProgressLevenshteinRatio(track_progress=False)
        
        result = levenshtein_metric.score(output=test_output, reference="The answer is 42")
        print(f"✅ Levenshtein metric test successful: {result.value:.3f}")
        
        print("\n🎉 All evaluation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run evaluation test"""
    print("🔍 MathPal Evaluation Test")
    print("=" * 50)
    
    success = test_evaluation()
    
    if success:
        print("\n✅ Evaluation test completed successfully!")
        print("🎯 You can now run: make evaluate-llm-compatible")
    else:
        print("\n❌ Evaluation test failed!")
        print("💡 Check the error messages above for details")

if __name__ == "__main__":
    main()
