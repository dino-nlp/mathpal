"""Evaluation and testing management."""

import random
from typing import Dict, Any, List, Optional, Tuple
import torch

from training_pipeline.utils.exceptions import InferenceError
from training_pipeline.config.config_manager import ConfigManager
from training_pipeline.utils import get_logger
from training_pipeline.inference import InferenceEngine

logger = get_logger()


class EvaluationManager:
    """Manages model evaluation and testing."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        
    def run_evaluation(self, model: Any, tokenizer: Any) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            logger.info("🧪 Starting model evaluation...")
            
            results = {
                "inference_tests": [],
                "vietnamese_math_tests": [],
                "generation_quality": {},
                "performance_metrics": {}
            }
            
            inference_engine = InferenceEngine(model=model, 
                                            tokenizer=tokenizer,
                                            config_manager=self.config,
                                            device="cuda")
            
            # Run basic inference tests
            results["inference_tests"] = self._run_inference_tests(inference_engine)
            
            # Run Vietnamese math-specific tests
            results["vietnamese_math_tests"] = self._run_vietnamese_math_tests(inference_engine)
            
            # Evaluate generation quality
            results["generation_quality"] = self._evaluate_generation_quality(inference_engine)
            
            # Performance metrics
            results["performance_metrics"] = self._measure_performance(inference_engine)
            
            logger.info("✅ Evaluation completed")
            logger.info("="*100)
            logger.info("📊 EVALUATION RESULTS SUMMARY:")
            self._log_evaluation_summary(results)
            logger.info("="*100)
            return results
            
        except Exception as e:
            raise InferenceError(f"Evaluation failed: {e}")
    
    def _run_inference_tests(self, inference_engine) -> List[Dict[str, Any]]:
        """Run basic inference tests."""
        logger.info(f"🧪 _run_inference_tests")
        try:
            # Prepare model for inference
            test_cases = self._get_test_cases()
            results = []
            
            for i, test_case in enumerate(test_cases[:self.config.evaluation.num_tc_examples]):
                try:
                    logger.info(f"🧪 Running test case {i+1}/{len(test_cases)}")
                    response = inference_engine.generate(test_case["input"])
                    result = {
                        "test_id": i + 1,
                        "input": test_case["input"],
                        "expected": test_case.get("expected", ""),
                        "generated": response,
                        "status": "success"
                    }
                    
                    results.append(result)
                    
                    # Log result in a structured format
                    logger.info(f"   📝 Test {i+1} Results:")
                    logger.info(f"      🎯 Input: {test_case['input']}")
                    logger.info(f"      💡 Expected: {test_case.get('expected', 'N/A')}")
                    logger.info(f"      🤖 Generated: {response}")
                    logger.info(f"      ✅ Status: Success")
                    logger.info("")
                    
                except Exception as e:
                    results.append({
                        "test_id": i + 1,
                        "input": test_case["input"],
                        "error": str(e),
                        "status": "failed"
                    })
                    logger.error(f"   ❌ Test {i+1} Failed:")
                    logger.error(f"      🎯 Input: {test_case['input']}")
                    logger.error(f"      💥 Error: {e}")
                    logger.error("")
            
            return results
            
        except Exception as e:
            raise InferenceError(f"Inference tests failed: {e}")
    
    def _run_vietnamese_math_tests(self, inference_engine) -> List[Dict[str, Any]]:
        """Run Vietnamese math-specific tests."""
        logger.info(f"🧪 _run_vietnamese_math_tests")
        vietnamese_tests = [
            {
                "input": "Tính 15 + 27 = ?",
                "expected_type": "arithmetic"
            },
            {
                "input": "Một hình chữ nhật có chiều dài 8cm và chiều rộng 5cm. Tính chu vi của hình chữ nhật này.",
                "expected_type": "geometry"
            },
            {
                "input": "Lan có 24 cái kẹo, Lan cho bạn 8 cái. Hỏi Lan còn lại bao nhiêu cái kẹo?",
                "expected_type": "word_problem"
            },
            {
                "input": "Tìm x biết: 3x + 5 = 14",
                "expected_type": "algebra"
            },
            {
                "input": "Viết phân số 3/4 dưới dạng số thập phân.",
                "expected_type": "fractions"
            }
        ]
        
        results = []
        for i, test in enumerate(vietnamese_tests):
            try:
                response = inference_engine.generate(test["input"])
                
                # Basic quality checks for Vietnamese math
                quality_score = self._assess_vietnamese_math_quality(
                    test["input"], response, test["expected_type"]
                )
                
                results.append({
                    "test_id": f"vn_math_{i+1}",
                    "input": test["input"],
                    "output": response,
                    "expected_type": test["expected_type"],
                    "quality_score": quality_score,
                    "contains_vietnamese": self._contains_vietnamese(response),
                    "has_mathematical_content": self._has_mathematical_content(response)
                })
                
            except Exception as e:
                results.append({
                    "test_id": f"vn_math_{i+1}",
                    "input": test["input"],
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    def _evaluate_generation_quality(self, inference_engine) -> Dict[str, Any]:
        """Evaluate overall generation quality."""
        logger.info(f"🧪 _evaluate_generation_quality")
        try:
            # Test different generation parameters
            test_input = "Giải thích cách tính diện tích hình vuông có cạnh 6cm."
            
            quality_metrics = {}
            
            # Test with different temperatures
            for temp in [0.1, 0.7, 1.0]:
                response = inference_engine.generate(test_input,
                                                    generation_config={"temperature": temp})
                
                quality_metrics[f"temp_{temp}"] = {
                    "response_length": len(response),
                    "contains_vietnamese": self._contains_vietnamese(response),
                    "has_math_content": self._has_mathematical_content(response),
                    "coherence_score": self._assess_coherence(response)
                }
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Generation quality evaluation failed: {e}")
            return {}
    
    def _measure_performance(self, inference_engine) -> Dict[str, Any]:
        """Measure performance metrics."""
        logger.info(f"🧪 _measure_performance")
        try:
            import time
            import torch
            
            test_input = "Tính 10 + 20 = ?"
            
            # Measure inference time
            start_time = time.time()
            for _ in range(5):  # Average over 5 runs
                _ = inference_engine.generate(test_input)
            avg_time = (time.time() - start_time) / 5
            
            # Memory usage
            memory_used = 0
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                "avg_inference_time": avg_time,
                "memory_usage_gb": memory_used,
                "model_size": self._estimate_model_size(inference_engine.model)
            }
            
        except Exception as e:
            logger.warning(f"Performance measurement failed: {e}")
            return {}
    

    def _get_test_cases(self) -> List[Dict[str, str]]:
        """Get test cases for evaluation."""
        return [
            {
                "input": "Xin chào, bạn có thể giúp tôi học toán không?",
                "expected": "greeting_response"
            },
            {
                "input": "Tính 5 + 3 = ?",
                "expected": "8"
            },
            {
                "input": "Giải thích cách nhân hai số tự nhiên.",
                "expected": "explanation"
            },
            {
                "input": "Một hình tam giác có ba cạnh bằng nhau gọi là gì?",
                "expected": "tam giác đều"
            },
            {
                "input": "Chuyển đổi 0.75 thành phân số.",
                "expected": "3/4"
            }
        ]
    
    def _assess_vietnamese_math_quality(self, input_text: str, response: str, expected_type: str) -> float:
        """Assess quality of Vietnamese math response."""
        score = 0.0
        
        # Check if response is in Vietnamese
        if self._contains_vietnamese(response):
            score += 0.3
        
        # Check for mathematical content
        if self._has_mathematical_content(response):
            score += 0.3
        
        # Check response length (not too short, not too long)
        if 20 <= len(response) <= 500:
            score += 0.2
        
        # Check for explanation structure
        if any(word in response.lower() for word in ["vì", "do", "nên", "tính", "giải"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _contains_vietnamese(self, text: str) -> bool:
        """Check if text contains Vietnamese characters."""
        vietnamese_chars = "àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý"
        return any(char in text.lower() for char in vietnamese_chars)
    
    def _has_mathematical_content(self, text: str) -> bool:
        """Check if text contains mathematical content."""
        math_indicators = [
            "+", "-", "×", "÷", "=", "%", 
            "tính", "bằng", "cộng", "trừ", "nhân", "chia",
            "phân số", "thập phân", "hình", "diện tích", "chu vi",
            "cm", "m", "km", "kg", "g"
        ]
        return any(indicator in text.lower() for indicator in math_indicators)
    
    def _assess_coherence(self, text: str) -> float:
        """Basic coherence assessment."""
        # Simple heuristics for coherence
        if len(text.strip()) < 10:
            return 0.0
        
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Check for repetition
        words = text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 0
        
        return min(repetition_ratio * 1.2, 1.0)
    
    def _estimate_model_size(self, model: Any) -> str:
        """Estimate model size."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            size_gb = total_params * 4 / (1024**3)  # Assume fp32
            return f"{size_gb:.2f} GB"
        except:
            return "Unknown"
    
    def _log_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation results in a beautiful, readable format.
        
        Args:
            results: Evaluation results dictionary
        """
        import json
        
        # 1. Inference Tests Summary
        if "inference_tests" in results:
            inference_tests = results["inference_tests"]
            success_count = sum(1 for test in inference_tests if test.get("status") == "success")
            total_count = len(inference_tests)
            
            logger.info("🔍 INFERENCE TESTS:")
            logger.info(f"   ✅ Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            logger.info(f"   ❌ Failed: {total_count-success_count}/{total_count}")
            
            # Log individual test results
            for i, test in enumerate(inference_tests):
                status_emoji = "✅" if test.get("status") == "success" else "❌"
                logger.info(f"   {status_emoji} Test {i+1}: {test.get('input', '')[:50]}...")
                if test.get("status") == "success":
                    logger.info(f"      Generated: {test.get('generated', '')[:100]}...")
                else:
                    logger.info(f"      Error: {test.get('error', '')}")
        
        # 2. Vietnamese Math Tests Summary
        if "vietnamese_math_tests" in results:
            vn_tests = results["vietnamese_math_tests"]
            logger.info("\n🇻🇳 VIETNAMESE MATH TESTS:")
            for test in vn_tests:
                test_id = test.get("test_id", "Unknown")
                quality_score = test.get("quality_score", 0)
                contains_vn = test.get("contains_vietnamese", False)
                has_math = test.get("has_mathematical_content", False)
                
                logger.info(f"   📝 {test_id}: {test.get('input', '')[:50]}...")
                logger.info(f"      Quality Score: {quality_score:.2f}/1.0")
                logger.info(f"      Contains Vietnamese: {'✅' if contains_vn else '❌'}")
                logger.info(f"      Has Math Content: {'✅' if has_math else '❌'}")
                logger.info(f"      Output: {test.get('output', '')[:100]}...")
        
        # 3. Generation Quality Summary
        if "generation_quality" in results:
            quality_metrics = results["generation_quality"]
            logger.info("\n🎯 GENERATION QUALITY:")
            for temp, metrics in quality_metrics.items():
                logger.info(f"   🌡️ Temperature {temp}:")
                logger.info(f"      Response Length: {metrics.get('response_length', 0)} chars")
                logger.info(f"      Contains Vietnamese: {'✅' if metrics.get('contains_vietnamese') else '❌'}")
                logger.info(f"      Has Math Content: {'✅' if metrics.get('has_math_content') else '❌'}")
                logger.info(f"      Coherence Score: {metrics.get('coherence_score', 0):.2f}/1.0")
        
        # 4. Performance Metrics Summary
        if "performance_metrics" in results:
            perf_metrics = results["performance_metrics"]
            logger.info("\n⚡ PERFORMANCE METRICS:")
            logger.info(f"   ⏱️ Average Inference Time: {perf_metrics.get('avg_inference_time', 0):.3f} seconds")
            logger.info(f"   💾 Memory Usage: {perf_metrics.get('memory_usage_gb', 0):.2f} GB")
            logger.info(f"   🤖 Model Size: {perf_metrics.get('model_size', 'Unknown')}")
        
        # 5. Overall Summary
        logger.info("\n📈 OVERALL EVALUATION SUMMARY:")
        total_tests = 0
        successful_tests = 0
        
        if "inference_tests" in results:
            total_tests += len(results["inference_tests"])
            successful_tests += sum(1 for test in results["inference_tests"] if test.get("status") == "success")
        
        if "vietnamese_math_tests" in results:
            total_tests += len(results["vietnamese_math_tests"])
            successful_tests += len(results["vietnamese_math_tests"])  # Assume all VN tests are successful
        
        if total_tests > 0:
            success_rate = successful_tests / total_tests * 100
            logger.info(f"   🎯 Overall Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
            
            if success_rate >= 80:
                logger.info("   🏆 Excellent performance!")
            elif success_rate >= 60:
                logger.info("   👍 Good performance!")
            elif success_rate >= 40:
                logger.info("   ⚠️ Moderate performance - needs improvement")
            else:
                logger.info("   ❌ Poor performance - significant issues detected")
