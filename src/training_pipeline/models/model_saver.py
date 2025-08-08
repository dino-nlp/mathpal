"""Model saving utilities with multiple format support."""

import os
from typing import Optional, Dict, Any, Union
from pathlib import Path


class ModelSaver:
    """Handles saving models in various formats."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize ModelSaver.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer/processor to save
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def save_lora_adapters(
        self,
        save_path: Union[str, Path],
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save only LoRA adapter weights.
        
        Args:
            save_path: Local path to save adapters
            push_to_hub: Whether to push to HuggingFace Hub
            repo_id: Repository ID for Hub upload
            token: HuggingFace token
            **kwargs: Additional arguments for saving
            
        Returns:
            Path where adapters were saved
        """
        save_path = str(save_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving LoRA adapters to {save_path}...")
        
        try:
            if push_to_hub and repo_id:
                # Save to HuggingFace Hub
                self.model.push_to_hub(repo_id, token=token, **kwargs)
                self.tokenizer.push_to_hub(repo_id, token=token, **kwargs)
                print(f"✅ LoRA adapters saved to Hub: {repo_id}")
                return repo_id
            else:
                # Save locally
                self.model.save_pretrained(save_path, **kwargs)
                self.tokenizer.save_pretrained(save_path, **kwargs)
                print(f"✅ LoRA adapters saved locally: {save_path}")
                return save_path
                
        except Exception as e:
            print(f"❌ Error saving LoRA adapters: {e}")
            raise
    
    def save_merged_model(
        self,
        save_path: Union[str, Path],
        precision: str = "fp16",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save merged model (base + adapters) in specified precision.
        
        Args:
            save_path: Local path to save model
            precision: Model precision ("fp16", "bf16", "fp32")
            push_to_hub: Whether to push to HuggingFace Hub
            repo_id: Repository ID for Hub upload  
            token: HuggingFace token
            **kwargs: Additional arguments for saving
            
        Returns:
            Path where model was saved
        """
        save_path = str(save_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving merged model ({precision}) to {save_path}...")
        
        # Map precision to save method
        save_method_map = {
            "fp16": "merged_16bit",
            "bf16": "merged_16bit", 
            "fp32": "merged_32bit"
        }
        
        save_method = save_method_map.get(precision, "merged_16bit")
        
        try:
            if push_to_hub and repo_id:
                # Save to HuggingFace Hub
                self.model.push_to_hub_merged(
                    repo_id, 
                    self.tokenizer,
                    save_method=save_method,
                    token=token,
                    **kwargs
                )
                print(f"✅ Merged model saved to Hub: {repo_id}")
                return repo_id
            else:
                # Save locally  
                self.model.save_pretrained_merged(
                    save_path,
                    self.tokenizer,
                    save_method=save_method,
                    **kwargs
                )
                print(f"✅ Merged model saved locally: {save_path}")
                return save_path
                
        except Exception as e:
            print(f"❌ Error saving merged model: {e}")
            raise
    
    def save_gguf_model(
        self,
        save_path: Union[str, Path],
        quantization: str = "q8_0",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save model in GGUF format for llama.cpp compatibility.
        
        Args:
            save_path: Local path to save model
            quantization: Quantization method ("q4_k_m", "q8_0", "f16", "bf16")
            push_to_hub: Whether to push to HuggingFace Hub
            repo_id: Repository ID for Hub upload
            token: HuggingFace token
            **kwargs: Additional arguments for saving
            
        Returns:
            Path where model was saved
        """
        save_path = str(save_path)
        
        print(f"Saving GGUF model ({quantization}) to {save_path}...")
        
        # Validate quantization method
        valid_quants = ["q4_k_m", "q8_0", "f16", "bf16"]
        if quantization not in valid_quants:
            raise ValueError(f"Invalid quantization: {quantization}. Valid options: {valid_quants}")
        
        try:
            if push_to_hub and repo_id:
                # Save to HuggingFace Hub
                self.model.push_to_hub_gguf(
                    repo_id,
                    self.tokenizer,
                    quantization_method=quantization,
                    token=token,
                    **kwargs
                )
                print(f"✅ GGUF model saved to Hub: {repo_id}")
                return repo_id
            else:
                # Save locally
                self.model.save_pretrained_gguf(
                    save_path,
                    self.tokenizer,
                    quantization_method=quantization,
                    **kwargs
                )
                print(f"✅ GGUF model saved locally: {save_path}")
                return save_path
                
        except Exception as e:
            print(f"❌ Error saving GGUF model: {e}")
            raise
    
    def save_all_formats(
        self,
        base_save_path: Union[str, Path],
        model_name: str,
        formats: Optional[Dict[str, Dict[str, Any]]] = None,
        push_to_hub: bool = False,
        hub_username: Optional[str] = None,
        token: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save model in multiple formats.
        
        Args:
            base_save_path: Base directory for saving
            model_name: Name for the model
            formats: Dictionary specifying formats and their configs
            push_to_hub: Whether to push to HuggingFace Hub
            hub_username: Username for Hub uploads
            token: HuggingFace token
            
        Returns:
            Dictionary mapping format names to save paths
        """
        if formats is None:
            formats = {
                "lora": {},
                "merged_fp16": {"precision": "fp16"},
                "gguf_q8": {"quantization": "q8_0"}
            }
        
        base_save_path = Path(base_save_path)
        results = {}
        
        for format_name, config in formats.items():
            try:
                if format_name.startswith("lora"):
                    save_path = base_save_path / f"{model_name}-lora"
                    repo_id = f"{hub_username}/{model_name}-lora" if hub_username else None
                    
                    results[format_name] = self.save_lora_adapters(
                        save_path=save_path,
                        push_to_hub=push_to_hub,
                        repo_id=repo_id,
                        token=token,
                        **config
                    )
                    
                elif format_name.startswith("merged"):
                    precision = config.get("precision", "fp16")
                    save_path = base_save_path / f"{model_name}-{precision}"
                    repo_id = f"{hub_username}/{model_name}-{precision}" if hub_username else None
                    
                    results[format_name] = self.save_merged_model(
                        save_path=save_path,
                        precision=precision,
                        push_to_hub=push_to_hub,
                        repo_id=repo_id,
                        token=token,
                        **{k: v for k, v in config.items() if k != "precision"}
                    )
                    
                elif format_name.startswith("gguf"):
                    quantization = config.get("quantization", "q8_0")
                    save_path = base_save_path / f"{model_name}-{quantization}"
                    repo_id = f"{hub_username}/{model_name}-{quantization}" if hub_username else None
                    
                    results[format_name] = self.save_gguf_model(
                        save_path=save_path,
                        quantization=quantization,
                        push_to_hub=push_to_hub,
                        repo_id=repo_id,
                        token=token,
                        **{k: v for k, v in config.items() if k != "quantization"}
                    )
                    
            except Exception as e:
                print(f"❌ Failed to save {format_name}: {e}")
                results[format_name] = f"Error: {e}"
        
        return results
    
    @staticmethod
    def get_available_formats() -> Dict[str, str]:
        """
        Get available save formats and their descriptions.
        
        Returns:
            Dictionary mapping format names to descriptions
        """
        return {
            "lora": "LoRA adapters only (small file size, requires base model)",
            "merged_fp16": "Full merged model in FP16 (for VLLM, TGI)",
            "merged_bf16": "Full merged model in BF16 (for modern hardware)",
            "merged_fp32": "Full merged model in FP32 (highest precision)",
            "gguf_f16": "GGUF format F16 (for llama.cpp)",
            "gguf_q8_0": "GGUF format Q8_0 (balanced size/quality)",
            "gguf_q4_k_m": "GGUF format Q4_K_M (smaller size, slight quality loss)"
        }