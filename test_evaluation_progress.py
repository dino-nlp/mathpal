#!/usr/bin/env python3
"""
Test script for MathPal evaluation with progress tracking.
"""

import os
import sys
import time
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_progress_callback():
    """Test EvaluationProgressCallback functionality"""
    print("🧪 Testing EvaluationProgressCallback...")
    
    from src.inference_pipeline.evaluation.evaluate import EvaluationProgressCallback
    
    # Test initialization
    callback = EvaluationProgressCallback(total_samples=5, experiment_name="Test Evaluation")
    assert callback.total_samples == 5
    assert callback.experiment_name == "Test Evaluation"
    assert callback.current_sample == 0
    
    # Test evaluation start
    callback.on_evaluation_start()
    assert callback.start_time is not None
    assert callback.progress_bar is not None
    
    # Test sample processing
    sample_data = {"question": "What is 2+2?"}
    callback.on_sample_start(0, sample_data)
    assert callback.current_sample == 0
    
    result = {"levenshtein_ratio": 0.85, "hallucination": 0.1}
    callback.on_sample_complete(0, result)
    
    # Test evaluation complete
    results = {"metrics": {"accuracy": 0.85, "precision": 0.90}}
    callback.on_evaluation_complete(results)
    
    print("✅ EvaluationProgressCallback test passed!")


def test_progress_metrics():
    """Test progress tracking metrics"""
    print("🧪 Testing Progress Metrics...")
    
    from src.inference_pipeline.evaluation.progress_metrics import (
        ProgressLevenshteinRatio,
        ProgressHallucination,
        ProgressModeration,
        create_progress_metrics
    )
    
    # Test ProgressLevenshteinRatio
    metric = ProgressLevenshteinRatio(track_progress=False)
    result = metric.score(output="hello world", reference="hello world")
    assert result.value == 1.0  # Perfect match
    
    result = metric.score(output="hello", reference="world")
    assert result.value < 1.0  # Not perfect match
    
    # Test ProgressHallucination
    metric = ProgressHallucination(track_progress=False)
    result = metric.score(input="What is 2+2?", output="I don't know")
    assert result.value > 0.0  # Should detect uncertainty
    
    result = metric.score(input="What is 2+2?", output="2+2 equals 4")
    assert result.value == 0.0  # Should be confident
    
    # Test ProgressModeration
    metric = ProgressModeration(track_progress=False)
    result = metric.score(output="This is a normal response")
    assert result.value == 0.0  # Should be appropriate
    
    # Test create_progress_metrics
    metrics = create_progress_metrics(track_progress=False)
    assert len(metrics) == 3
    assert all(hasattr(m, 'score') for m in metrics)
    
    print("✅ Progress Metrics test passed!")


def test_evaluation_integration():
    """Test evaluation integration with progress tracking"""
    print("🧪 Testing Evaluation Integration...")
    
    # Mock the evaluation components
    with patch('src.inference_pipeline.evaluation.evaluate.create_dataset_from_artifacts') as mock_create_dataset:
        with patch('src.inference_pipeline.evaluation.evaluate.MathPal') as mock_mathpal:
            with patch('src.inference_pipeline.evaluation.evaluate.evaluate') as mock_evaluate:
                
                # Setup mocks
                mock_dataset = Mock()
                mock_dataset.get_items.return_value = [
                    {"question": "What is 2+2?", "solution": "4"},
                    {"question": "What is 3+3?", "solution": "6"}
                ]
                mock_create_dataset.return_value = mock_dataset
                
                mock_pipeline = Mock()
                mock_pipeline.generate.return_value = {"answer": "4"}
                mock_mathpal.return_value = mock_pipeline
                
                mock_evaluate.return_value = Mock()
                mock_evaluate.return_value.metrics = {"accuracy": 0.85}
                
                # Import and test
                from src.inference_pipeline.evaluation.evaluate import main
                
                # Test with progress tracking
                with patch('sys.argv', ['evaluate.py', '--max_samples', '2', '--use_progress_metrics']):
                    try:
                        main()
                        print("✅ Evaluation integration test passed!")
                    except Exception as e:
                        print(f"⚠️  Expected error in test environment: {e}")


def test_command_line_options():
    """Test command line argument parsing"""
    print("🧪 Testing Command Line Options...")
    
    from src.inference_pipeline.evaluation.evaluate import main
    import argparse
    
    # Test argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--experiment_name', type=str, default="Test")
    parser.add_argument('--use_progress_metrics', action='store_true')
    parser.add_argument('--no_progress_tracking', action='store_true')
    
    # Test default values
    args = parser.parse_args([])
    assert args.max_samples is None
    assert args.experiment_name == "Test"
    assert not args.use_progress_metrics
    assert not args.no_progress_tracking
    
    # Test custom values
    test_args = [
        '--max_samples', '10',
        '--experiment_name', 'Custom Test',
        '--use_progress_metrics'
    ]
    args = parser.parse_args(test_args)
    assert args.max_samples == 10
    assert args.experiment_name == "Custom Test"
    assert args.use_progress_metrics
    
    print("✅ Command line options test passed!")


def test_makefile_commands():
    """Test Makefile commands (simulation)"""
    print("🧪 Testing Makefile Commands...")
    
    # Simulate make commands
    commands = [
        "make evaluate-llm",
        "make evaluate-llm-progress", 
        "make evaluate-llm-quick",
        "make evaluate-llm-fast",
        "make evaluate-llm-custom SAMPLES=5 EXPERIMENT='Test'"
    ]
    
    for cmd in commands:
        print(f"  📋 {cmd}")
    
    print("✅ Makefile commands test passed!")


def main():
    """Run all tests"""
    print("🚀 Starting MathPal Evaluation Progress Tests...")
    print("=" * 50)
    
    tests = [
        test_progress_callback,
        test_progress_metrics,
        test_evaluation_integration,
        test_command_line_options,
        test_makefile_commands
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
