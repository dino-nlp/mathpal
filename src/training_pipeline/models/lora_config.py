"""LoRA configuration management."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class LoRAConfigManager:
    """Manages LoRA configuration for different model architectures."""
    
    @staticmethod
    def get_gemma_target_modules() -> List[str]:
        """Get default target modules for Gemma models."""
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    @staticmethod
    def get_llama_target_modules() -> List[str]:
        """Get default target modules for LLaMA models."""
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    @staticmethod
    def get_mistral_target_modules() -> List[str]:
        """Get default target modules for Mistral models."""
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    @classmethod
    def get_target_modules_for_model(cls, model_name: str) -> Optional[List[str]]:
        """
        Get target modules based on model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of target modules or None for auto-detection
        """
        model_name_lower = model_name.lower()
        
        if "gemma" in model_name_lower:
            return cls.get_gemma_target_modules()
        elif "llama" in model_name_lower:
            return cls.get_llama_target_modules()  
        elif "mistral" in model_name_lower:
            return cls.get_mistral_target_modules()
        else:
            # Let Unsloth auto-detect
            return None
    
    @staticmethod
    def create_lora_config(
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        target_modules: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create LoRA configuration dictionary.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            bias: Bias configuration ("none", "all", "lora_only")
            target_modules: Target modules for LoRA
            model_name: Model name for auto target module detection
            use_rslora: Whether to use RSLoRA
            use_dora: Whether to use DoRA
            **kwargs: Additional configuration parameters
            
        Returns:
            LoRA configuration dictionary
        """
        config = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "use_rslora": use_rslora,
            "use_dora": use_dora,
        }
        
        # Auto-detect target modules if not provided
        if target_modules is None and model_name:
            target_modules = LoRAConfigManager.get_target_modules_for_model(model_name)
        
        if target_modules:
            config["target_modules"] = target_modules
        
        # Add any additional kwargs
        config.update(kwargs)
        
        return config
    
    @staticmethod
    def get_recommended_configs() -> Dict[str, Dict[str, Any]]:
        """
        Get recommended LoRA configurations for different scenarios.
        
        Returns:
            Dictionary of configuration presets
        """
        return {
            "lightweight": {
                "r": 8,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "bias": "none",
                "description": "Lightweight config for quick experiments"
            },
            "balanced": {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none", 
                "description": "Balanced config for most use cases"
            },
            "high_capacity": {
                "r": 64,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "description": "High capacity config for complex tasks"
            },
            "memory_efficient": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "bias": "none",
                "description": "Memory efficient config for limited resources"
            }
        }
    
    @staticmethod
    def validate_lora_config(config: Dict[str, Any]) -> None:
        """
        Validate LoRA configuration.
        
        Args:
            config: LoRA configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["r", "lora_alpha"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required LoRA config key: {key}")
        
        if config["r"] <= 0:
            raise ValueError("LoRA rank (r) must be positive")
        
        if config["lora_alpha"] <= 0:
            raise ValueError("LoRA alpha must be positive")
        
        if "lora_dropout" in config and not (0 <= config["lora_dropout"] <= 1):
            raise ValueError("LoRA dropout must be between 0 and 1")
        
        if "bias" in config and config["bias"] not in ["none", "all", "lora_only"]:
            raise ValueError("LoRA bias must be 'none', 'all', or 'lora_only'")
    
    @staticmethod
    def print_config_info(config: Dict[str, Any]) -> None:
        """
        Print information about LoRA configuration.
        
        Args:
            config: LoRA configuration to display
        """
        print("\n🔧 LoRA Configuration:")
        print(f"   Rank (r): {config.get('r', 'Not set')}")
        print(f"   Alpha: {config.get('lora_alpha', 'Not set')}")
        print(f"   Dropout: {config.get('lora_dropout', 'Not set')}")
        print(f"   Bias: {config.get('bias', 'Not set')}")
        print(f"   Target modules: {config.get('target_modules', 'Auto-detect')}")
        print(f"   Use RSLoRA: {config.get('use_rslora', False)}")
        print(f"   Use DoRA: {config.get('use_dora', False)}")
        
        # Calculate expected parameter reduction
        r = config.get('r', 16)
        if 'target_modules' in config:
            num_modules = len(config['target_modules'])
            print(f"   Estimated modules affected: {num_modules}")
            print(f"   Parameter efficiency: ~{100 / (r + 1):.1f}x reduction per module")