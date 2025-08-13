"""Device and hardware utilities."""

import torch
from typing import Dict, Any, Optional, List


class DeviceUtils:
    """Utilities for device management and hardware detection."""
    
    @staticmethod
    def get_device() -> str:
        """
        Get the best available device.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get comprehensive device information.
        
        Returns:
            Dictionary with device information
        """
        info = {
            "device": DeviceUtils.get_device(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        }
        
        # CUDA information
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_compute_capability": torch.cuda.get_device_capability(),
            })
            
            # Memory information
            memory_info = DeviceUtils.get_cuda_memory_info()
            info.update(memory_info)
        
        return info
    
    @staticmethod
    def get_cuda_memory_info(device: Optional[int] = None) -> Dict[str, float]:
        """
        Get CUDA memory information.
        
        Args:
            device: Optional device index
            
        Returns:
            Dictionary with memory information in GB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        if device is None:
            device = torch.cuda.current_device()
        
        try:
            props = torch.cuda.get_device_properties(device)
            
            return {
                "total_memory_gb": props.total_memory / 1024**3,
                "allocated_memory_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "reserved_memory_gb": torch.cuda.memory_reserved(device) / 1024**3,
                "free_memory_gb": (props.total_memory - torch.cuda.memory_reserved(device)) / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def print_device_info() -> None:
        """Print device information in a formatted way."""
        info = DeviceUtils.get_device_info()
        
        print("\n🖥️ Device Information:")
        print(f"   Device: {info['device']}")
        print(f"   PyTorch version: {info['torch_version']}")
        print(f"   CUDA available: {info['cuda_available']}")
        print(f"   MPS available: {info['mps_available']}")
        
        if info['cuda_available']:
            print(f"\n🔥 CUDA Information:")
            print(f"   CUDA version: {info['cuda_version']}")
            print(f"   cuDNN version: {info['cudnn_version']}")
            print(f"   Device count: {info['cuda_device_count']}")
            print(f"   Current device: {info['cuda_current_device']}")
            print(f"   Device name: {info['cuda_device_name']}")
            print(f"   Compute capability: {info['cuda_compute_capability']}")
            
            if 'total_memory_gb' in info:
                print(f"\n💾 Memory Information:")
                print(f"   Total memory: {info['total_memory_gb']:.2f} GB")
                print(f"   Allocated memory: {info['allocated_memory_gb']:.2f} GB")
                print(f"   Reserved memory: {info['reserved_memory_gb']:.2f} GB")
                print(f"   Free memory: {info['free_memory_gb']:.2f} GB")
    
    @staticmethod
    def get_gpu_memory_gb(device: Optional[int] = None) -> float:
        """
        Get total GPU memory in GB.
        
        Args:
            device: Optional device index
            
        Returns:
            Total GPU memory in GB, or 0 if CUDA not available
        """
        if not torch.cuda.is_available():
            return 0.0
        
        if device is None:
            device = torch.cuda.current_device()
        
        try:
            props = torch.cuda.get_device_properties(device)
            return props.total_memory / 1024**3
        except Exception:
            return 0.0
    
    @staticmethod
    def clear_cuda_cache() -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 CUDA cache cleared")
        else:
            print("⚠️ CUDA not available, cache not cleared")
    
    @staticmethod
    def reset_cuda_peak_stats() -> None:
        """Reset CUDA peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print("📊 CUDA peak memory stats reset")
        else:
            print("⚠️ CUDA not available, stats not reset")
    
    @staticmethod
    def set_cuda_device(device_id: int) -> None:
        """
        Set CUDA device.
        
        Args:
            device_id: Device ID to set
        """
        if torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                print(f"🔧 CUDA device set to: {device_id}")
            else:
                print(f"❌ Invalid device ID: {device_id}")
        else:
            print("⚠️ CUDA not available")
    
    @staticmethod
    def get_optimal_batch_size(
        model: torch.nn.Module,
        input_shape: tuple,
        max_memory_gb: float = 10.0,
        start_batch_size: int = 1
    ) -> int:
        """
        Find optimal batch size based on available memory.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (excluding batch dimension)
            max_memory_gb: Maximum memory to use in GB
            start_batch_size: Starting batch size for testing
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, returning default batch size")
            return start_batch_size
        
        device = DeviceUtils.get_device()
        model = model.to(device)
        model.eval()
        
        batch_size = start_batch_size
        max_batch_size = start_batch_size
        
        print(f"🔍 Finding optimal batch size (max memory: {max_memory_gb:.1f}GB)...")
        
        try:
            while True:
                try:
                    # Create dummy input
                    dummy_input = torch.randn(batch_size, *input_shape).to(device)
                    
                    # Forward pass
                    with torch.no_grad():
                        _ = model(dummy_input)
                    
                    # Check memory usage
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    
                    if memory_used > max_memory_gb:
                        break
                    
                    max_batch_size = batch_size
                    batch_size *= 2
                    
                    # Clear cache
                    del dummy_input
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    else:
                        raise
                
                # Safety limit
                if batch_size > 1024:
                    break
            
            print(f"✅ Optimal batch size: {max_batch_size}")
            return max_batch_size
            
        except Exception as e:
            print(f"❌ Error finding optimal batch size: {e}")
            return start_batch_size
        finally:
            torch.cuda.empty_cache()
    
    @staticmethod
    def benchmark_device(num_operations: int = 1000) -> Dict[str, float]:
        """
        Benchmark device performance.
        
        Args:
            num_operations: Number of operations to benchmark
            
        Returns:
            Benchmark results
        """
        device = DeviceUtils.get_device()
        
        print(f"🏃‍♂️ Benchmarking {device} performance...")
        
        # Matrix multiplication benchmark
        import time
        
        size = 1000
        x = torch.randn(size, size).to(device)
        y = torch.randn(size, size).to(device)
        
        # Warmup
        for _ in range(10):
            _ = torch.mm(x, y)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_operations):
            _ = torch.mm(x, y)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        ops_per_second = num_operations / total_time
        
        results = {
            "device": device,
            "total_time": total_time,
            "operations_per_second": ops_per_second,
            "time_per_operation": total_time / num_operations
        }
        
        print(f"📊 Benchmark results:")
        print(f"   Operations per second: {ops_per_second:.2f}")
        print(f"   Time per operation: {results['time_per_operation']*1000:.2f}ms")
        
        return results
    
    @staticmethod
    def monitor_memory_usage(func, *args, **kwargs):
        """
        Monitor memory usage during function execution.
        
        Args:
            func: Function to monitor
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result and memory stats
        """
        if not torch.cuda.is_available():
            result = func(*args, **kwargs)
            return result, {"message": "CUDA not available"}
        
        # Reset stats
        torch.cuda.reset_peak_memory_stats()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory stats
        final_memory = torch.cuda.memory_allocated() / 1024**3
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        memory_stats = {
            "initial_memory_gb": initial_memory,
            "final_memory_gb": final_memory,
            "peak_memory_gb": peak_memory,
            "memory_delta_gb": final_memory - initial_memory,
            "peak_delta_gb": peak_memory - initial_memory,
        }
        
        print(f"💾 Memory usage:")
        print(f"   Peak: {peak_memory:.2f}GB")
        print(f"   Delta: {memory_stats['memory_delta_gb']:.2f}GB")
        
        return result, memory_stats