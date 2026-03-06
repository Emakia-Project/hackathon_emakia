"""
Classifier component for the Emakia Validator Agent.

This module provides content classification functionality using AI models.
"""

import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

from src.wrappers import OpenAIWrapper, FireworksWrapper, GPTOSSWrapper, LlamaWrapper


class Classifier:
    """
    Content classifier that uses AI models to categorize content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.classification_config = config.get('classification', {})
        self.model_wrappers = {}
        
        # Initialize model wrappers
        self._initialize_wrappers()
        
        logger.info("Classifier initialized successfully")
    
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
    
    async def classify(self, content: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Classify content using AI models.
        
        Args:
            content: Content to classify
            content_type: Type of content (text, image, etc.)
            
        Returns:
            Classification results
        """
        try:
            logger.info(f"Starting classification for {content_type} content")
            
            # Get classification categories
            categories = self._get_categories(content_type)
            
            # Get default model provider
            default_provider = self.config.get('models', {}).get('default', 'openai')
            
            if default_provider not in self.model_wrappers:
                logger.warning(f"Default provider {default_provider} not available, trying fallback")
                # Try to find any available provider
                available_providers = list(self.model_wrappers.keys())
                if not available_providers:
                    return {
                        'category': 'unknown',
                        'confidence': 0.0,
                        'reasoning': 'No AI models available for classification',
                        'all_categories': categories,
                        'model': 'none',
                        'classification_type': 'fallback'
                    }
                default_provider = available_providers[0]
            
            # Perform classification
            wrapper = self.model_wrappers[default_provider]
            result = await wrapper.classify(content, categories)
            
            # Add classification type and timestamp
            result['classification_type'] = 'ai_model'
            result['model_provider'] = default_provider
            result['timestamp'] = asyncio.get_event_loop().time()
            
            # Apply confidence thresholds
            result = self._apply_confidence_thresholds(result)
            
            logger.info("Classification completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            raise
    
    def _get_categories(self, content_type: str) -> List[str]:
        """
        Get classification categories for specific content type.
        
        Args:
            content_type: Type of content
            
        Returns:
            List of categories
        """
        base_categories = self.classification_config.get('categories', [
            'safe', 'unsafe', 'inappropriate', 'spam', 'hate_speech', 
            'violence', 'adult_content', 'misinformation'
        ])
        
        if content_type == "text":
            return base_categories
        elif content_type == "image":
            return [
                'safe', 'unsafe', 'inappropriate', 'explicit_content', 
                'violence', 'adult_content', 'spam'
            ]
        elif content_type == "video":
            return [
                'safe', 'unsafe', 'inappropriate', 'explicit_content', 
                'violence', 'adult_content', 'spam', 'copyright_violation'
            ]
        else:
            return base_categories
    
    def _apply_confidence_thresholds(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply confidence thresholds to classification results.
        
        Args:
            result: Classification result
            
        Returns:
            Result with applied thresholds
        """
        category = result.get('category')
        confidence = result.get('confidence', 0.0)
        
        # Get confidence threshold for the category
        thresholds = self.classification_config.get('confidence_thresholds', {})
        threshold = thresholds.get(category, 0.7)  # Default threshold
        
        # Check if confidence meets threshold
        meets_threshold = confidence >= threshold
        
        # Update result
        result['meets_threshold'] = meets_threshold
        result['threshold'] = threshold
        result['threshold_met'] = meets_threshold
        
        # If confidence is too low, mark as uncertain
        if not meets_threshold:
            result['category'] = 'uncertain'
            result['reasoning'] = f"Confidence {confidence:.2f} below threshold {threshold}"
        
        return result
    
    async def classify_batch(self, contents: List[str], content_type: str = "text") -> List[Dict[str, Any]]:
        """
        Classify multiple content items in batch.
        
        Args:
            contents: List of content items to classify
            content_type: Type of content
            
        Returns:
            List of classification results
        """
        logger.info(f"Starting batch classification of {len(contents)} items")
        
        tasks = [
            self.classify(content, content_type) 
            for content in contents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error classifying item {i}: {str(result)}")
            else:
                valid_results.append(result)
        
        logger.info(f"Batch classification completed. {len(valid_results)}/{len(contents)} successful")
        return valid_results
    
    async def classify_with_multiple_models(self, content: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Classify content using multiple AI models and combine results.
        
        Args:
            content: Content to classify
            content_type: Type of content
            
        Returns:
            Combined classification results
        """
        logger.info(f"Starting multi-model classification for {content_type} content")
        
        categories = self._get_categories(content_type)
        results = []
        
        # Classify with each available model
        for provider, wrapper in self.model_wrappers.items():
            try:
                result = await wrapper.classify(content, categories)
                result['model_provider'] = provider
                results.append(result)
            except Exception as e:
                logger.error(f"Classification failed with {provider}: {str(e)}")
        
        if not results:
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'reasoning': 'All models failed to classify',
                'all_categories': categories,
                'model': 'none',
                'classification_type': 'multi_model_fallback'
            }
        
        # Combine results
        combined_result = self._combine_classification_results(results)
        combined_result['classification_type'] = 'multi_model'
        combined_result['timestamp'] = asyncio.get_event_loop().time()
        
        logger.info("Multi-model classification completed successfully")
        return combined_result
    
    def _combine_classification_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine classification results from multiple models.
        
        Args:
            results: List of classification results
            
        Returns:
            Combined classification result
        """
        if not results:
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'reasoning': 'No results to combine',
                'all_categories': [],
                'model': 'none'
            }
        
        # Count categories
        category_counts = {}
        total_confidence = 0.0
        all_reasonings = []
        all_categories = set()
        
        for result in results:
            category = result.get('category', 'unknown')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')
            categories = result.get('all_categories', [])
            
            category_counts[category] = category_counts.get(category, 0) + 1
            total_confidence += confidence
            all_reasonings.append(reasoning)
            all_categories.update(categories)
        
        # Find most common category
        most_common_category = max(category_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(results)
        
        # Combine reasonings
        combined_reasoning = f"Multi-model consensus: {most_common_category} (confidence: {avg_confidence:.2f})"
        
        return {
            'category': most_common_category,
            'confidence': avg_confidence,
            'reasoning': combined_reasoning,
            'all_categories': list(all_categories),
            'model': 'multi_model',
            'model_results': results,
            'category_distribution': category_counts
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on classifier.
        
        Returns:
            Health status information
        """
        health_status = {
            'status': 'healthy',
            'component': 'classifier',
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
    
    def get_available_categories(self, content_type: str = "text") -> List[str]:
        """
        Get available classification categories for a content type.
        
        Args:
            content_type: Type of content
            
        Returns:
            List of available categories
        """
        return self._get_categories(content_type)
    
    def get_confidence_thresholds(self) -> Dict[str, float]:
        """
        Get confidence thresholds for all categories.
        
        Returns:
            Dictionary of category thresholds
        """
        return self.classification_config.get('confidence_thresholds', {})
