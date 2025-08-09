"""Training utilities and helpers."""

import os
import time
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: Optional[float] = None
    step: Optional[int] = None
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    training_time: Optional[float] = None
    samples_per_second: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class TrainingUtils:
    """Utilities for training management and monitoring."""
    
    @staticmethod
    def setup_output_directory(output_dir: str) -> str:
        """
        Setup output directory for training.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Created output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        return output_dir
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get device and memory information.
        
        Returns:
            Dictionary with device information
        """
        info = {
            "device": "cpu",
            "device_count": 0,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                "device": "cuda",
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info
    
    @staticmethod
    def print_device_info() -> None:
        """Print device information."""
        info = TrainingUtils.get_device_info()
        
        print(f"\n🖥️ Device Information:")
        print(f"   Device: {info['device']}")
        print(f"   CUDA available: {info['cuda_available']}")
        
        if info['cuda_available']:
            print(f"   Device count: {info['device_count']}")
            print(f"   Current device: {info['current_device']}")
            print(f"   Device name: {info['device_name']}")
            print(f"   Memory allocated: {info['memory_allocated_gb']:.2f} GB")
            print(f"   Memory reserved: {info['memory_reserved_gb']:.2f} GB")
            print(f"   Memory total: {info['memory_total_gb']:.2f} GB")
    
    @staticmethod
    def clear_cuda_cache() -> None:
        """Clear CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 CUDA cache cleared")
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        print(f"🎲 Random seed set to {seed}")
    
    @staticmethod
    def monitor_training_progress(
        trainer,
        log_interval: int = 100,
        save_interval: int = 500
    ) -> None:
        """
        Monitor training progress with custom logging.
        
        Args:
            trainer: SFTTrainer instance
            log_interval: Steps between progress logs
            save_interval: Steps between model saves
        """
        # This would be implemented as a callback or hook
        # For now, just print the configuration
        print(f"📊 Monitoring setup:")
        print(f"   Log interval: {log_interval} steps")
        print(f"   Save interval: {save_interval} steps")
    
    @staticmethod
    def estimate_training_time(
        num_samples: int,
        batch_size: int,
        max_steps: int,
        estimated_seconds_per_step: float = 1.0
    ) -> Dict[str, float]:
        """
        Estimate training time.
        
        Args:
            num_samples: Number of training samples
            batch_size: Effective batch size
            max_steps: Maximum training steps
            estimated_seconds_per_step: Estimated time per step
            
        Returns:
            Dictionary with time estimates
        """
        steps_per_epoch = num_samples / batch_size
        total_epochs = max_steps / steps_per_epoch
        total_seconds = max_steps * estimated_seconds_per_step
        
        return {
            "steps_per_epoch": steps_per_epoch,
            "total_epochs": total_epochs,
            "estimated_total_seconds": total_seconds,
            "estimated_total_minutes": total_seconds / 60,
            "estimated_total_hours": total_seconds / 3600
        }
    
    @staticmethod
    def print_training_estimates(
        num_samples: int,
        batch_size: int,
        max_steps: int,
        estimated_seconds_per_step: float = 1.0
    ) -> None:
        """Print training time estimates."""
        estimates = TrainingUtils.estimate_training_time(
            num_samples, batch_size, max_steps, estimated_seconds_per_step
        )
        
        print(f"\n⏱️ Training Estimates:")
        print(f"   Steps per epoch: {estimates['steps_per_epoch']:.1f}")
        print(f"   Total epochs: {estimates['total_epochs']:.2f}")
        print(f"   Estimated time: {estimates['estimated_total_hours']:.2f} hours")
    
    @staticmethod
    def save_training_config(config_dict: Dict[str, Any], save_path: str) -> None:
        """
        Save training configuration to file.
        
        Args:
            config_dict: Configuration dictionary
            save_path: Path to save configuration
        """
        import json
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Training config saved to {save_path}")
    
    @staticmethod
    def create_training_summary(
        trainer_stats: Any,
        config: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create training summary from trainer statistics.
        
        Args:
            trainer_stats: Training statistics from trainer
            config: Training configuration
            save_path: Optional path to save summary
            
        Returns:
            Training summary dictionary
        """
        summary = {
            "training_completed": True,
            "config": config,
            "stats": trainer_stats.__dict__ if hasattr(trainer_stats, '__dict__') else str(trainer_stats),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if save_path:
            TrainingUtils.save_training_config(summary, save_path)
        
        return summary
    
    @staticmethod
    def log_memory_usage(step: Optional[int] = None) -> None:
        """
        Log current memory usage.
        
        Args:
            step: Optional training step number
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            step_info = f"Step {step}: " if step is not None else ""
            print(f"🔍 {step_info}Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    @staticmethod
    def cleanup_training_artifacts(
        output_dir: str,
        keep_checkpoints: int = 3,
        keep_logs: bool = True
    ) -> None:
        """
        Cleanup training artifacts, keeping only recent checkpoints.
        
        Args:
            output_dir: Output directory path
            keep_checkpoints: Number of recent checkpoints to keep
            keep_logs: Whether to keep log files
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return
        
        # Find checkpoint directories
        checkpoint_dirs = [d for d in output_path.iterdir() 
                          if d.is_dir() and d.name.startswith('checkpoint-')]
        
        if len(checkpoint_dirs) > keep_checkpoints:
            # Sort by modification time and keep only recent ones
            checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_checkpoint in checkpoint_dirs[keep_checkpoints:]:
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    print(f"🗑️ Removed old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    print(f"⚠️ Could not remove {old_checkpoint}: {e}")
        
        print(f"🧹 Cleanup completed, kept {min(len(checkpoint_dirs), keep_checkpoints)} checkpoints")