"""
OpenAI model wrapper for the Emakia Validator Agent.

This module provides a wrapper for OpenAI's API, implementing the base wrapper interface.
"""

import asyncio
from typing import Dict, Any, List
from openai import AsyncOpenAI
from loguru import logger

from .base_wrapper import BaseModelWrapper


class OpenAIWrapper(BaseModelWrapper):
    """
    Wrapper for OpenAI models.
    
    This class provides a unified interface for interacting with OpenAI's
    GPT models through their API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI wrapper.
        
        Args:
            config: Configuration dictionary for OpenAI
        """
        super().__init__(config)
        
        # Initialize OpenAI client
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info("OpenAI wrapper initialized successfully")
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using OpenAI's API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response with metadata
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Get parameters
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Make API call
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                ),
                timeout=self.timeout
            )
            
            # Extract response
            content = response.choices[0].message.content
            usage = response.usage
            
            return {
                'content': content,
                'model': self.model_name,
                'usage': {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason
            }
            
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API request timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    async def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """
        Classify text into predefined categories using OpenAI.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            
        Returns:
            Classification results
        """
        # Create classification prompt
        categories_str = ", ".join(categories)
        prompt = f"""
        Classify the following text into one of these categories: {categories_str}
        
        Text: {text}
        
        Please respond with a JSON object containing:
        1. "category": the most appropriate category
        2. "confidence": confidence score between 0 and 1
        3. "reasoning": brief explanation for the classification
        
        Response:
        """
        
        try:
            response = await self.generate(prompt, temperature=0.1)
            
            # Parse response (assuming JSON format)
            import json
            try:
                result = json.loads(response['content'])
                return {
                    'category': result.get('category'),
                    'confidence': result.get('confidence', 0.0),
                    'reasoning': result.get('reasoning', ''),
                    'all_categories': categories,
                    'model': self.model_name
                }
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                return {
                    'category': 'unknown',
                    'confidence': 0.0,
                    'reasoning': 'Failed to parse model response',
                    'all_categories': categories,
                    'model': self.model_name,
                    'raw_response': response['content']
                }
                
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            raise
    
    async def validate(self, content: str, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate content against specified rules using OpenAI.
        
        Args:
            content: Content to validate
            validation_rules: Rules to validate against
            
        Returns:
            Validation results
        """
        # Create validation prompt
        rules_str = "\n".join([f"- {rule}: {desc}" for rule, desc in validation_rules.items()])
        prompt = f"""
        Validate the following content against these rules:
        
        {rules_str}
        
        Content: {content}
        
        Please respond with a JSON object containing:
        1. "is_valid": boolean indicating if content passes all rules
        2. "violations": list of rule violations found
        3. "confidence": confidence score between 0 and 1
        4. "suggestions": suggestions for improvement
        
        Response:
        """
        
        try:
            response = await self.generate(prompt, temperature=0.1)
            
            # Parse response
            import json
            try:
                result = json.loads(response['content'])
                return {
                    'is_valid': result.get('is_valid', False),
                    'violations': result.get('violations', []),
                    'confidence': result.get('confidence', 0.0),
                    'suggestions': result.get('suggestions', []),
                    'model': self.model_name,
                    'rules_checked': list(validation_rules.keys())
                }
            except json.JSONDecodeError:
                return {
                    'is_valid': False,
                    'violations': ['Failed to parse validation response'],
                    'confidence': 0.0,
                    'suggestions': ['Check model response format'],
                    'model': self.model_name,
                    'rules_checked': list(validation_rules.keys()),
                    'raw_response': response['content']
                }
                
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on OpenAI wrapper.
        
        Returns:
            Health status information
        """
        base_health = super().health_check()
        
        # Add OpenAI-specific health info
        base_health.update({
            'api_key_configured': bool(self.config.get('api_key')),
            'client_initialized': hasattr(self, 'client')
        })
        
        return base_health
