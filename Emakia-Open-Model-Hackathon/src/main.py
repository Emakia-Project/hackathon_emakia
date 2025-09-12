#!/usr/bin/env python3
"""
Main entry point for the Emakia Validator Agent.

This module provides the core functionality for content validation and classification
using multiple AI model providers and validation pipelines.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from loguru import logger

from src.pipeline.validator import Validator
from src.pipeline.classifier import Classifier
from src.pipeline.output_normalizer import OutputNormalizer
from src.utils.logging import setup_logging
from src.utils.metrics import MetricsCollector
from src.config.model_config import load_config


class EmakiaValidatorAgent:
    """
    Main agent class that orchestrates validation and classification tasks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Emakia Validator Agent.
        
        Args:
            config_path: Path to configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        setup_logging()
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.validator = Validator(self.config)
        self.classifier = Classifier(self.config)
        self.normalizer = OutputNormalizer()
        self.metrics = MetricsCollector()
        
        logger.info("Emakia Validator Agent initialized successfully")
    
    async def validate_content(self, content: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Validate content using the validation pipeline.
        
        Args:
            content: The content to validate
            content_type: Type of content (text, image, etc.)
            
        Returns:
            Validation results
        """
        try:
            logger.info(f"Starting validation for {content_type} content")
            
            # Validate content
            validation_result = await self.validator.validate(content, content_type)
            
            # Classify content
            classification_result = await self.classifier.classify(content, content_type)
            
            # Normalize output
            normalized_result = self.normalizer.normalize({
                'validation': validation_result,
                'classification': classification_result
            })
            
            # Record metrics
            self.metrics.record_validation(content_type, normalized_result)
            
            logger.info("Content validation completed successfully")
            return normalized_result
            
        except Exception as e:
            logger.error(f"Error during content validation: {str(e)}")
            self.metrics.record_error(str(e))
            raise
    
    async def batch_validate(self, contents: list, content_type: str = "text") -> list:
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
            self.validate_content(content, content_type) 
            for content in contents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing item {i}: {str(result)}")
            else:
                valid_results.append(result)
        
        logger.info(f"Batch validation completed. {len(valid_results)}/{len(contents)} successful")
        return valid_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics and statistics.
        
        Returns:
            Metrics data
        """
        return self.metrics.get_summary()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Health status
        """
        health_status = {
            'status': 'healthy',
            'components': {}
        }
        
        try:
            # Check validator
            health_status['components']['validator'] = self.validator.health_check()
            
            # Check classifier
            health_status['components']['classifier'] = self.classifier.health_check()
            
            # Check normalizer
            health_status['components']['normalizer'] = self.normalizer.health_check()
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            logger.error(f"Health check failed: {str(e)}")
        
        return health_status


async def main():
    """
    Main function for CLI usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Emakia Validator Agent")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--content", help="Content to validate")
    parser.add_argument("--type", default="text", help="Content type")
    parser.add_argument("--batch", help="Path to file with multiple content items")
    parser.add_argument("--health", action="store_true", help="Perform health check")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = EmakiaValidatorAgent(args.config)
    
    if args.health:
        # Perform health check
        health = agent.health_check()
        print(f"Health Status: {health}")
        return
    
    if args.content:
        # Validate single content
        result = await agent.validate_content(args.content, args.type)
        print(f"Validation Result: {result}")
    
    elif args.batch:
        # Batch validation
        with open(args.batch, 'r') as f:
            contents = [line.strip() for line in f if line.strip()]
        
        results = await agent.batch_validate(contents, args.type)
        print(f"Batch Results: {results}")
    
    else:
        # Interactive mode
        print("Emakia Validator Agent - Interactive Mode")
        print("Enter content to validate (or 'quit' to exit):")
        
        while True:
            try:
                content = input("> ")
                if content.lower() == 'quit':
                    break
                
                result = await agent.validate_content(content)
                print(f"Result: {result}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
