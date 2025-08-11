"""Checkpoint and model saving management."""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core.exceptions import CheckpointError
from ..core.enhanced_config import ComprehensiveTrainingConfig
from ..utils import get_logger

logger = get_logger()


class CheckpointManager:
    """Manages model checkpoints and saving."""
    
    def __init__(self, config: ComprehensiveTrainingConfig):
        self.config = config
        self.saved_models = {}
        
    def save_model(self, model: Any, tokenizer: Any, training_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Save model in requested formats.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            training_results: Optional training results
            
        Returns:
            Dictionary mapping format to save path
        """
        try:
            logger.info("💾 Saving model in requested formats...")
            
            save_results = {}
            base_path = self.config.get_output_dir()
            model_name = f"gemma3n-{self.config.output.experiment_name}"
            
            for format_name in self.config.output.save_formats:
                try:
                    save_path = self._save_format(model, tokenizer, format_name, base_path, model_name)
                    save_results[format_name] = save_path
                    logger.info(f"   ✅ {format_name}: {save_path}")
                except Exception as e:
                    error_msg = f"Error saving {format_name}: {e}"
                    save_results[format_name] = error_msg
                    logger.error(f"   ❌ {error_msg}")
            
            # Save training metadata
            self._save_training_metadata(training_results, base_path)
            
            self.saved_models = save_results
            return save_results
            
        except Exception as e:
            raise CheckpointError(f"Failed to save model: {e}")
    
    def _save_format(self, model: Any, tokenizer: Any, format_name: str, base_path: str, model_name: str) -> str:
        """Save model in specific format."""
        
        if format_name == "lora":
            return self._save_lora(model, tokenizer, base_path, model_name)
        elif format_name == "merged_16bit":
            return self._save_merged_16bit(model, tokenizer, base_path, model_name)
        elif format_name == "merged_4bit":
            return self._save_merged_4bit(model, tokenizer, base_path, model_name)
        elif format_name.startswith("gguf_"):
            quantization = format_name.split("_")[1]
            return self._save_gguf(model, tokenizer, base_path, model_name, quantization)
        else:
            raise CheckpointError(f"Unsupported save format: {format_name}")
    
    def _save_lora(self, model: Any, tokenizer: Any, base_path: str, model_name: str) -> str:
        """Save LoRA adapters only."""
        save_path = os.path.join(base_path, f"{model_name}-lora")
        
        # Save using standard methods
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        return save_path
    
    def _save_merged_16bit(self, model: Any, tokenizer: Any, base_path: str, model_name: str) -> str:
        """Save merged model in 16-bit precision."""
        save_path = os.path.join(base_path, f"{model_name}-merged-16bit")
        
        # Check if model has Unsloth save method
        if hasattr(model, 'save_pretrained_merged'):
            model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
        else:
            # Fallback to standard saving
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        
        return save_path
    
    def _save_merged_4bit(self, model: Any, tokenizer: Any, base_path: str, model_name: str) -> str:
        """Save merged model in 4-bit precision."""
        save_path = os.path.join(base_path, f"{model_name}-merged-4bit")
        
        if hasattr(model, 'save_pretrained_merged'):
            model.save_pretrained_merged(save_path, tokenizer, save_method="merged_4bit")
        else:
            raise CheckpointError("4-bit saving requires Unsloth model")
        
        return save_path
    
    def _save_gguf(self, model: Any, tokenizer: Any, base_path: str, model_name: str, quantization: str) -> str:
        """Save model in GGUF format."""
        save_path = os.path.join(base_path, f"{model_name}-{quantization}.gguf")
        
        if hasattr(model, 'save_pretrained_gguf'):
            model.save_pretrained_gguf(save_path, tokenizer, quantization_method=quantization)
        else:
            raise CheckpointError("GGUF saving requires Unsloth model")
        
        return save_path
    
    def _save_training_metadata(self, training_results: Optional[Dict[str, Any]], base_path: str) -> None:
        """Save training metadata."""
        try:
            metadata = {
                "config": self.config.to_dict(),
                "training_results": training_results,
                "saved_formats": self.config.output.save_formats,
            }
            
            metadata_path = os.path.join(base_path, "training_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📋 Training metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training metadata: {e}")
    
    def push_to_hub(self, model: Any, tokenizer: Any) -> Optional[str]:
        """Push model to HuggingFace Hub."""
        try:
            if not self.config.hub.push_to_hub:
                return None
            
            username = self.config.hub.username or os.getenv("HF_USERNAME")
            if not username:
                logger.warning("⚠️ HuggingFace username not provided, skipping Hub push")
                return None
            
            repo_name = self.config.hub.repo_name or self.config.output.experiment_name
            hub_repo = f"{username}/{repo_name}"
            
            logger.info(f"📤 Pushing model to HuggingFace Hub: {hub_repo}")
            
            # Get token
            token = self.config.hub.token or os.getenv("HF_TOKEN")
            
            if hasattr(model, 'push_to_hub_merged'):
                # Use Unsloth's optimized push method
                model.push_to_hub_merged(
                    hub_repo,
                    tokenizer,
                    save_method="lora",
                    token=token,
                    private=self.config.hub.private
                )
            else:
                # Standard HuggingFace push
                model.push_to_hub(hub_repo, token=token, private=self.config.hub.private)
                tokenizer.push_to_hub(hub_repo, token=token, private=self.config.hub.private)
            
            logger.info(f"✅ Model pushed to Hub: {hub_repo}")
            return hub_repo
            
        except Exception as e:
            logger.error(f"❌ Failed to push to Hub: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        try:
            if not os.path.exists(checkpoint_path):
                raise CheckpointError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load training metadata
            metadata_path = os.path.join(checkpoint_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata
            else:
                logger.warning("No training metadata found in checkpoint")
                return {}
                
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        try:
            base_dir = self.config.output.base_dir
            if not os.path.exists(base_dir):
                return []
            
            checkpoints = []
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid checkpoint
                    if any(os.path.exists(os.path.join(item_path, f)) 
                          for f in ["config.json", "pytorch_model.bin", "model.safetensors"]):
                        checkpoints.append(item_path)
            
            return sorted(checkpoints)
            
        except Exception as e:
            logger.warning(f"Failed to list checkpoints: {e}")
            return []
    
    def cleanup_old_checkpoints(self) -> None:
        """Cleanup old checkpoints based on save_total_limit."""
        try:
            if self.config.output.save_total_limit <= 0:
                return
            
            checkpoints = self.list_checkpoints()
            if len(checkpoints) <= self.config.output.save_total_limit:
                return
            
            # Sort by modification time and keep only the newest ones
            checkpoints_with_time = [
                (cp, os.path.getmtime(cp)) for cp in checkpoints
            ]
            checkpoints_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old checkpoints
            to_remove = checkpoints_with_time[self.config.output.save_total_limit:]
            
            for checkpoint_path, _ in to_remove:
                try:
                    import shutil
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"🗑️ Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoints: {e}")
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the best checkpoint based on metrics."""
        # This would require implementing metric tracking
        # For now, return the most recent checkpoint
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None
