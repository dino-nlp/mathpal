#!/usr/bin/env python3
"""
Demo script for MathPal evaluation with progress tracking.
This script demonstrates the new progress tracking features.
"""

import os
import sys
import time
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_progress_callback():
    """Demo the EvaluationProgressCallback"""
    print("🚀 Demo: EvaluationProgressCallback")
    print("=" * 40)
    
    try:
        from src.inference_pipeline.evaluation.evaluate import EvaluationProgressCallback
        
        # Create callback
        callback = EvaluationProgressCallback(
            total_samples=5, 
            experiment_name="Demo Evaluation"
        )
        
        # Start evaluation
        callback.on_evaluation_start()
        
        # Simulate processing samples
        sample_data = [
            {"question": "What is 2+2?"},
            {"question": "What is 3+3?"},
            {"question": "What is 4+4?"},
            {"question": "What is 5+5?"},
            {"question": "What is 6+6?"}
        ]
        
        for i, data in enumerate(sample_data):
            callback.on_sample_start(i, data)
            time.sleep(0.5)  # Simulate processing time
            
            result = {
                "levenshtein_ratio": 0.8 + (i * 0.05),
                "hallucination": 0.1 - (i * 0.02),
                "moderation": 0.05
            }
            callback.on_sample_complete(i, result)
        
        # Complete evaluation
        results = {
            "metrics": {
                "levenshtein_ratio": 0.85,
                "hallucination": 0.12,
                "moderation": 0.05,
                "style": 0.78
            }
        }
        callback.on_evaluation_complete(results)
        
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_progress_metrics():
    """Demo the progress tracking metrics"""
    print("\n🚀 Demo: Progress Tracking Metrics")
    print("=" * 40)
    
    try:
        from src.inference_pipeline.evaluation.progress_metrics import (
            ProgressLevenshteinRatio,
            ProgressHallucination,
            ProgressModeration,
            create_progress_metrics
        )
        
        # Test Levenshtein ratio
        print("📊 Testing ProgressLevenshteinRatio...")
        metric = ProgressLevenshteinRatio(track_progress=False)
        result = metric.score(output="hello world", reference="hello world")
        print(f"   Perfect match: {result.value:.3f}")
        
        result = metric.score(output="hello", reference="world")
        print(f"   Different strings: {result.value:.3f}")
        
        # Test Hallucination detection
        print("\n📊 Testing ProgressHallucination...")
        metric = ProgressHallucination(track_progress=False)
        result = metric.score(input="What is 2+2?", output="I don't know")
        print(f"   Uncertainty response: {result.value:.3f}")
        
        result = metric.score(input="What is 2+2?", output="2+2 equals 4")
        print(f"   Confident response: {result.value:.3f}")
        
        # Test Moderation
        print("\n📊 Testing ProgressModeration...")
        metric = ProgressModeration(track_progress=False)
        result = metric.score(output="This is a normal response")
        print(f"   Normal content: {result.value:.3f}")
        
        # Test metric creation
        print("\n📊 Testing metric creation...")
        metrics = create_progress_metrics(track_progress=False)
        print(f"   Created {len(metrics)} metrics")
        
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_command_line_interface():
    """Demo the command line interface"""
    print("\n🚀 Demo: Command Line Interface")
    print("=" * 40)
    
    print("Available commands:")
    print("  make evaluate-llm                    # Standard evaluation")
    print("  make evaluate-llm-progress          # Evaluation with progress tracking")
    print("  make evaluate-llm-quick             # Quick evaluation (5 samples)")
    print("  make evaluate-llm-fast              # Fast evaluation (no progress)")
    print("  make evaluate-llm-custom SAMPLES=10 EXPERIMENT='Test'")
    print()
    print("Direct Python commands:")
    print("  python -m src.inference_pipeline.evaluation.evaluate --help")
    print("  python -m src.inference_pipeline.evaluation.evaluate --max_samples 5 --use_progress_metrics")
    print("  python -m src.inference_pipeline.evaluation.evaluate --no_progress_tracking")
    
    print("✅ Demo completed successfully!")


def demo_features_summary():
    """Show a summary of new features"""
    print("\n🚀 Demo: Features Summary")
    print("=" * 40)
    
    features = [
        "📊 Progress Bar với tqdm",
        "⏱️  Real-time progress tracking",
        "📈 Detailed performance metrics",
        "🔧 Custom progress metrics",
        "⚡ Fast mode (no progress tracking)",
        "🎯 Flexible sample limits",
        "📋 Comprehensive logging",
        "🛠️  Easy command line interface"
    ]
    
    print("New Features:")
    for feature in features:
        print(f"  {feature}")
    
    print("\nBenefits:")
    print("  🚀 Faster development cycles")
    print("  📊 Better visibility into evaluation progress")
    print("  🔍 Easier debugging and monitoring")
    print("  ⚡ Flexible performance options")
    print("  🎯 Better user experience")
    
    print("✅ Demo completed successfully!")


def main():
    """Run all demos"""
    print("🎉 MathPal Evaluation Progress Demo")
    print("=" * 50)
    
    demos = [
        demo_progress_callback,
        demo_progress_metrics,
        demo_command_line_interface,
        demo_features_summary
    ]
    
    for demo in demos:
        try:
            demo()
            print()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print()
    
    print("🎉 All demos completed!")
    print("\n📚 Next steps:")
    print("  1. Run: make evaluate-llm-quick")
    print("  2. Run: make evaluate-llm-progress")
    print("  3. Check the README.md for more details")
    print("  4. Explore the Opik dashboard for results")


if __name__ == "__main__":
    main()
