"""
Provider factory for the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time

from ..config import ConfigManager
from ..utils import (
    ProviderError,
    get_logger
)


class BaseProvider(ABC):
    """
    Base class for all LLM providers in the evaluation pipeline.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize base provider.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using the provider.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        pass


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter provider for LLM access.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize OpenRouter provider.
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        self.openrouter_config = config.get_openrouter_config()
        
        # Initialize OpenRouter client (placeholder)
        self.client = None
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        
        self.logger.info("OpenRouter provider initialized")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using OpenRouter.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Raises:
            ProviderError: If generation fails
        """
        # Rate limiting
        self._check_rate_limit()
        
        try:
            # This is a placeholder implementation
            # In real implementation, this would use the actual OpenRouter API
            
            model = kwargs.get("model", self.openrouter_config.models["primary"])
            
            self.logger.debug(f"Generating with OpenRouter model: {model}")
            
            # Placeholder response
            response = f"OpenRouter response to: {prompt[:50]}..."
            
            # Update rate limiting
            self._update_rate_limit()
            
            return response
            
        except Exception as e:
            raise ProviderError(f"Error generating with OpenRouter: {e}")
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def _check_rate_limit(self):
        """Check rate limiting."""
        current_time = time.time()
        
        # Check requests per minute
        if current_time - self.last_request_time < 60:
            if self.request_count >= self.openrouter_config.rate_limits["requests_per_minute"]:
                sleep_time = 60 - (current_time - self.last_request_time)
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.request_count = 0
        else:
            self.request_count = 0
    
    def _update_rate_limit(self):
        """Update rate limiting counters."""
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _initialize_client(self):
        """
        Initialize OpenRouter client.
        """
        # Placeholder for OpenRouter client initialization
        self.logger.info("Initializing OpenRouter client (placeholder)")
        return None


class FallbackProvider(BaseProvider):
    """
    Fallback provider for when primary provider fails.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize fallback provider.
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        self.logger.info("Fallback provider initialized")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using fallback method.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        self.logger.warning("Using fallback provider")
        
        # Simple fallback response
        return f"Fallback response: {prompt[:100]}..."
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class ProviderFactory:
    """
    Factory for creating provider instances.
    """
    
    @staticmethod
    def create_provider(config: ConfigManager, provider_type: str = "openrouter") -> BaseProvider:
        """
        Create a provider instance.
        
        Args:
            config: Configuration manager
            provider_type: Type of provider to create
            
        Returns:
            Provider instance
            
        Raises:
            ProviderError: If provider type is not supported
        """
        if provider_type.lower() == "openrouter":
            return OpenRouterProvider(config)
        elif provider_type.lower() == "fallback":
            return FallbackProvider(config)
        else:
            raise ProviderError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_supported_providers() -> list:
        """
        Get list of supported provider types.
        
        Returns:
            List of supported provider types
        """
        return ["openrouter", "fallback"]
