"""
Validator component for the Emakia Validator Agent.

This module provides content validation functionality using AI models and rule-based checks.
"""

import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

from src.wrappers import OpenAIWrapper, FireworksWrapper, GPTOSSWrapper, LlamaWrapper
from src.config.model_config import get_model_config


class Validator:
    """
    Content validator that uses AI models to validate content against various rules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config.get('validation', {})
        self.model_wrappers = {}
        
        # Initialize model wrappers
        self._initialize_wrappers()
        
        logger.info("Validator initialized successfully")
    
    def _initialize_wrappers(self):
        """
        Initialize model wrappers based on configuration.
        """
        models_config = self.config.get('models', {})
        providers = models_config.get('providers', {})
        
        for provider_name, provider_config in providers.items():
            try:
                if provider_name == 'openai':
                    self.model_wrappers[provider_name] = OpenAIWrapper(provider_config)
                elif provider_name == 'fireworks':
                    self.model_wrappers[provider_name] = FireworksWrapper(provider_config)
                elif provider_name == 'gpt_oss':
                    self.model_wrappers[provider_name] = GPTOSSWrapper(provider_config)
                elif provider_name == 'llama':
                    self.model_wrappers[provider_name] = LlamaWrapper(provider_config)
                else:
                    logger.warning(f"Unknown provider: {provider_name}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} wrapper: {str(e)}")
    
    async def validate(self, content: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Validate content using AI models and rule-based checks.
        
        Args:
            content: Content to validate
            content_type: Type of content (text, image, etc.)
            
        Returns:
            Validation results
        """
        try:
            logger.info(f"Starting validation for {content_type} content")
            
            # Basic content validation
            basic_validation = self._validate_basic_rules(content, content_type)
            
            # AI-based validation
            ai_validation = await self._validate_with_ai(content, content_type)
            
            # Combine results
            combined_result = self._combine_validation_results(basic_validation, ai_validation)
            
            logger.info("Validation completed successfully")
            return combined_result
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise
    
    def _validate_basic_rules(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Apply basic rule-based validation.
        
        Args:
            content: Content to validate
            content_type: Type of content
            
        Returns:
            Basic validation results
        """
        violations = []
        suggestions = []
        
        # Get content type specific rules
        content_rules = self.validation_config.get('content_types', {}).get(content_type, {})
        
        if content_type == "text":
            # Check length
            min_length = content_rules.get('min_length', 10)
            max_length = content_rules.get('max_length', 10000)
            
            if len(content) < min_length:
                violations.append(f"Content too short (minimum {min_length} characters)")
                suggestions.append("Add more content to meet minimum length requirement")
            
            if len(content) > max_length:
                violations.append(f"Content too long (maximum {max_length} characters)")
                suggestions.append("Reduce content length to meet maximum requirement")
            
            # Check for empty content
            if not content.strip():
                violations.append("Content is empty or contains only whitespace")
                suggestions.append("Provide meaningful content")
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'suggestions': suggestions,
            'confidence': 1.0 if len(violations) == 0 else 0.0,
            'validation_type': 'basic_rules'
        }
    
    async def _validate_with_ai(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Validate content using AI models.
        
        Args:
            content: Content to validate
            content_type: Type of content
            
        Returns:
            AI validation results
        """
        # Define validation rules based on content type
        validation_rules = self._get_validation_rules(content_type)
        
        # Get default model provider
        default_provider = self.config.get('models', {}).get('default', 'openai')
        
        if default_provider not in self.model_wrappers:
            logger.warning(f"Default provider {default_provider} not available, trying fallback")
            # Try to find any available provider
            available_providers = list(self.model_wrappers.keys())
            if not available_providers:
                return {
                    'is_valid': False,
                    'violations': ['No AI models available for validation'],
                    'suggestions': ['Check model configuration'],
                    'confidence': 0.0,
                    'validation_type': 'ai_fallback'
                }
            default_provider = available_providers[0]
        
        try:
            wrapper = self.model_wrappers[default_provider]
            result = await wrapper.validate(content, validation_rules)
            
            # Add validation type
            result['validation_type'] = 'ai_model'
            result['model_provider'] = default_provider
            
            return result
            
        except Exception as e:
            logger.error(f"AI validation failed: {str(e)}")
            return {
                'is_valid': False,
                'violations': [f'AI validation failed: {str(e)}'],
                'suggestions': ['Try again later or use different model'],
                'confidence': 0.0,
                'validation_type': 'ai_error',
                'model_provider': default_provider
            }
    
    def _get_validation_rules(self, content_type: str) -> Dict[str, str]:
        """
        Get validation rules for specific content type.
        
        Args:
            content_type: Type of content
            
        Returns:
            Validation rules dictionary
        """
        base_rules = {
            "no_hate_speech": "Content should not contain hate speech or discriminatory language",
            "no_violence": "Content should not promote or glorify violence",
            "no_harmful_content": "Content should not be harmful to individuals or groups",
            "appropriate_language": "Content should use appropriate and respectful language",
            "no_spam": "Content should not be spam or misleading"
        }
        
        if content_type == "text":
            return base_rules
        elif content_type == "image":
            return {
                **base_rules,
                "no_explicit_content": "Image should not contain explicit or inappropriate content",
                "no_violence": "Image should not contain violent or graphic content"
            }
        elif content_type == "video":
            return {
                **base_rules,
                "no_explicit_content": "Video should not contain explicit or inappropriate content",
                "no_violence": "Video should not contain violent or graphic content",
                "appropriate_duration": "Video should be of appropriate length"
            }
        else:
            return base_rules
    
    def _combine_validation_results(self, basic_result: Dict[str, Any], ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine basic and AI validation results.
        
        Args:
            basic_result: Basic validation results
            ai_result: AI validation results
            
        Returns:
            Combined validation results
        """
        # Combine violations
        all_violations = basic_result.get('violations', []) + ai_result.get('violations', [])
        
        # Combine suggestions
        all_suggestions = basic_result.get('suggestions', []) + ai_result.get('suggestions', [])
        
        # Determine overall validity
        basic_valid = basic_result.get('is_valid', False)
        ai_valid = ai_result.get('is_valid', False)
        
        # Content is valid only if both basic and AI validation pass
        overall_valid = basic_valid and ai_valid
        
        # Calculate combined confidence (weighted average)
        basic_confidence = basic_result.get('confidence', 0.0)
        ai_confidence = ai_result.get('confidence', 0.0)
        
        # Weight AI validation more heavily
        combined_confidence = (basic_confidence * 0.3) + (ai_confidence * 0.7)
        
        return {
            'is_valid': overall_valid,
            'violations': all_violations,
            'suggestions': all_suggestions,
            'confidence': combined_confidence,
            'validation_details': {
                'basic_validation': basic_result,
                'ai_validation': ai_result
            },
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on validator.
        
        Returns:
            Health status information
        """
        health_status = {
            'status': 'healthy',
            'component': 'validator',
            'model_wrappers': {}
        }
        
        try:
            for provider, wrapper in self.model_wrappers.items():
                health_status['model_wrappers'][provider] = wrapper.health_check()
            
            # Check if any wrappers are available
            if not self.model_wrappers:
                health_status['status'] = 'warning'
                health_status['message'] = 'No model wrappers available'
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    async def validate_batch(self, contents: List[str], content_type: str = "text") -> List[Dict[str, Any]]:
        """
        Validate multiple content items in batch.
        
        Args:
            contents: List of content items to validate
            content_type: Type of content
            
        Returns:
            List of validation results
        """
        logger.info(f"Starting batch validation of {len(contents)} items")
        
        tasks = [
            self.validate(content, content_type) 
            for content in contents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error validating item {i}: {str(result)}")
            else:
                valid_results.append(result)
        
        logger.info(f"Batch validation completed. {len(valid_results)}/{len(contents)} successful")
        return valid_results
