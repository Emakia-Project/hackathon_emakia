"""
Fireworks AI model wrapper for the Emakia Validator Agent.

This module provides a wrapper for Fireworks AI's API, implementing the base wrapper interface.
"""

import asyncio
from typing import Dict, Any, List
import aiohttp
from loguru import logger

from .base_wrapper import BaseModelWrapper


class FireworksWrapper(BaseModelWrapper):
    """
    Wrapper for Fireworks AI models.
    
    This class provides a unified interface for interacting with Fireworks AI's
    models through their API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Fireworks wrapper.
        
        Args:
            config: Configuration dictionary for Fireworks AI
        """
        super().__init__(config)
        
        # Initialize Fireworks configuration
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Fireworks API key is required")
        
        self.base_url = "https://api.fireworks.ai/inference/v1"
        self.session = None
        
        logger.info("Fireworks wrapper initialized successfully")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session.
        
        Returns:
            aiohttp ClientSession
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using Fireworks AI's API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response with metadata
        """
        try:
            session = await self._get_session()
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "temperature": kwargs.get('temperature', self.temperature),
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Extract response
                content = data['choices'][0]['message']['content']
                usage = data.get('usage', {})
                
                return {
                    'content': content,
                    'model': self.model_name,
                    'usage': {
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'completion_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0)
                    },
                    'finish_reason': data['choices'][0].get('finish_reason', 'stop')
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Fireworks API request timed out after {self.timeout} seconds")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Error calling Fireworks API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Fireworks API call: {str(e)}")
            raise
    
    async def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """
        Classify text into predefined categories using Fireworks AI.
        
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
        Validate content against specified rules using Fireworks AI.
        
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
    
    async def close(self):
        """
        Close the aiohttp session.
        """
        if self.session and not self.session.closed:
            await self.session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Fireworks wrapper.
        
        Returns:
            Health status information
        """
        base_health = super().health_check()
        
        # Add Fireworks-specific health info
        base_health.update({
            'api_key_configured': bool(self.api_key),
            'session_active': self.session is not None and not self.session.closed
        })
        
        return base_health
