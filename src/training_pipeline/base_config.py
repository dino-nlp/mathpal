"""Base configuration classes."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os
import yaml
import json


@dataclass
class BaseConfig(ABC):
    """Base configuration class for all configurations."""
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "BaseConfig":
        """Load config from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def save_json(self, json_path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @abstractmethod
    def validate(self) -> None:
        """Validate configuration."""
        pass


def get_env_var(var_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional default and required check."""
    value = os.getenv(var_name, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {var_name} is not set")
    return value