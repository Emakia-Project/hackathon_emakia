"""
Base model wrapper for the Emakia Validator Agent.

This module defines the base interface that all model wrappers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
from loguru import logger


class BaseModelWrapper(ABC):
    """
    Base class for all model wrappers.
    
    This class defines the interface that all model wrappers must implement
    to provide unified access to different AI model providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base wrapper.
        
        Args:
            config: Configuration dictionary for the model provider
        """
        self.config = config
        self.model_name = config.get('default_model', '')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.1)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.model_name}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        pass
    
    @abstractmethod
    async def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """
        Classify text into predefined categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            
        Returns:
            Classification results with confidence scores
        """
        pass
    
    @abstractmethod
    async def validate(self, content: str, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate content against specified rules.
        
        Args:
            content: Content to validate
            validation_rules: Rules to validate against
            
        Returns:
            Validation results
        """
        pass
    
    async def generate_with_retry(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate response with automatic retry logic.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await self.generate(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"All {self.max_retries} attempts failed")
        raise last_exception
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model wrapper.
        
        Returns:
            Health status information
        """
        return {
            'status': 'healthy',
            'provider': self.__class__.__name__,
            'model': self.model_name,
            'config_loaded': bool(self.config)
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model names
        """
        return self.config.get('models', [])
    
    def set_model(self, model_name: str) -> None:
        """
        Set the model to use for generation.
        
        Args:
            model_name: Name of the model to use
        """
        if model_name in self.get_available_models():
            self.model_name = model_name
            logger.info(f"Switched to model: {model_name}")
        else:
            raise ValueError(f"Model {model_name} not available. Available models: {self.get_available_models()}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        return {
            'name': self.model_name,
            'provider': self.__class__.__name__,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout
        }
