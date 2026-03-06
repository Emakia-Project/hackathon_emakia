"""
Output normalizer for the Emakia Validator Agent.

This module provides functionality to normalize and standardize outputs from
the validation and classification pipelines.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


class OutputNormalizer:
    """
    Normalizes and standardizes pipeline outputs.
    """
    
    def __init__(self):
        """
        Initialize the output normalizer.
        """
        self.standard_format = {
            'status': 'success',
            'timestamp': None,
            'content_type': None,
            'validation': {},
            'classification': {},
            'metadata': {},
            'errors': []
        }
        
        logger.info("Output normalizer initialized")
    
    def normalize(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize pipeline output to standard format.
        
        Args:
            pipeline_output: Raw pipeline output
            
        Returns:
            Normalized output
        """
        try:
            normalized = self.standard_format.copy()
            
            # Set timestamp
            normalized['timestamp'] = datetime.now().isoformat()
            
            # Extract validation results
            if 'validation' in pipeline_output:
                normalized['validation'] = self._normalize_validation(pipeline_output['validation'])
            
            # Extract classification results
            if 'classification' in pipeline_output:
                normalized['classification'] = self._normalize_classification(pipeline_output['classification'])
            
            # Extract metadata
            if 'metadata' in pipeline_output:
                normalized['metadata'] = pipeline_output['metadata']
            
            # Set content type if available
            if 'content_type' in pipeline_output:
                normalized['content_type'] = pipeline_output['content_type']
            
            # Handle errors
            if 'errors' in pipeline_output:
                normalized['errors'] = pipeline_output['errors']
                normalized['status'] = 'error'
            
            # Determine overall status
            normalized['status'] = self._determine_overall_status(normalized)
            
            logger.debug("Output normalized successfully")
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing output: {str(e)}")
            return self._create_error_output(str(e))
    
    def _normalize_validation(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize validation results.
        
        Args:
            validation_result: Raw validation result
            
        Returns:
            Normalized validation result
        """
        normalized = {
            'is_valid': False,
            'confidence': 0.0,
            'violations': [],
            'suggestions': [],
            'validation_type': 'unknown',
            'model_provider': 'unknown',
            'details': {}
        }
        
        # Copy basic fields
        for key in ['is_valid', 'confidence', 'violations', 'suggestions']:
            if key in validation_result:
                normalized[key] = validation_result[key]
        
        # Handle validation type
        if 'validation_type' in validation_result:
            normalized['validation_type'] = validation_result['validation_type']
        
        # Handle model provider
        if 'model_provider' in validation_result:
            normalized['model_provider'] = validation_result['model_provider']
        
        # Handle detailed validation results
        if 'validation_details' in validation_result:
            normalized['details'] = validation_result['validation_details']
        
        # Ensure violations and suggestions are lists
        if not isinstance(normalized['violations'], list):
            normalized['violations'] = [normalized['violations']] if normalized['violations'] else []
        
        if not isinstance(normalized['suggestions'], list):
            normalized['suggestions'] = [normalized['suggestions']] if normalized['suggestions'] else []
        
        return normalized
    
    def _normalize_classification(self, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize classification results.
        
        Args:
            classification_result: Raw classification result
            
        Returns:
            Normalized classification result
        """
        normalized = {
            'category': 'unknown',
            'confidence': 0.0,
            'reasoning': '',
            'all_categories': [],
            'model_provider': 'unknown',
            'classification_type': 'unknown',
            'threshold_met': False,
            'threshold': 0.0
        }
        
        # Copy basic fields
        for key in ['category', 'confidence', 'reasoning', 'all_categories']:
            if key in classification_result:
                normalized[key] = classification_result[key]
        
        # Handle model provider
        if 'model_provider' in classification_result:
            normalized['model_provider'] = classification_result['model_provider']
        
        # Handle classification type
        if 'classification_type' in classification_result:
            normalized['classification_type'] = classification_result['classification_type']
        
        # Handle threshold information
        if 'threshold_met' in classification_result:
            normalized['threshold_met'] = classification_result['threshold_met']
        
        if 'threshold' in classification_result:
            normalized['threshold'] = classification_result['threshold']
        
        # Ensure all_categories is a list
        if not isinstance(normalized['all_categories'], list):
            normalized['all_categories'] = [normalized['all_categories']] if normalized['all_categories'] else []
        
        return normalized
    
    def _determine_overall_status(self, normalized_output: Dict[str, Any]) -> str:
        """
        Determine overall status based on validation and classification results.
        
        Args:
            normalized_output: Normalized output
            
        Returns:
            Overall status
        """
        # Check for errors first
        if normalized_output.get('errors'):
            return 'error'
        
        # Check validation status
        validation = normalized_output.get('validation', {})
        if not validation.get('is_valid', True):
            return 'validation_failed'
        
        # Check classification status
        classification = normalized_output.get('classification', {})
        if classification.get('category') in ['unsafe', 'inappropriate', 'hate_speech', 'violence', 'adult_content']:
            return 'content_flagged'
        
        return 'success'
    
    def _create_error_output(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error output.
        
        Args:
            error_message: Error message
            
        Returns:
            Error output
        """
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'content_type': None,
            'validation': {
                'is_valid': False,
                'confidence': 0.0,
                'violations': ['Processing error occurred'],
                'suggestions': ['Try again later or contact support'],
                'validation_type': 'error',
                'model_provider': 'none',
                'details': {}
            },
            'classification': {
                'category': 'unknown',
                'confidence': 0.0,
                'reasoning': 'Classification failed due to processing error',
                'all_categories': [],
                'model_provider': 'none',
                'classification_type': 'error',
                'threshold_met': False,
                'threshold': 0.0
            },
            'metadata': {},
            'errors': [error_message]
        }
    
    def format_for_api(self, normalized_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format normalized output for API response.
        
        Args:
            normalized_output: Normalized output
            
        Returns:
            API-formatted output
        """
        api_output = {
            'success': normalized_output['status'] == 'success',
            'status': normalized_output['status'],
            'timestamp': normalized_output['timestamp'],
            'data': {
                'validation': normalized_output['validation'],
                'classification': normalized_output['classification']
            }
        }
        
        # Add metadata if present
        if normalized_output.get('metadata'):
            api_output['data']['metadata'] = normalized_output['metadata']
        
        # Add errors if present
        if normalized_output.get('errors'):
            api_output['errors'] = normalized_output['errors']
        
        return api_output
    
    def format_for_dashboard(self, normalized_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format normalized output for dashboard display.
        
        Args:
            normalized_output: Normalized output
            
        Returns:
            Dashboard-formatted output
        """
        dashboard_output = {
            'status': normalized_output['status'],
            'timestamp': normalized_output['timestamp'],
            'content_type': normalized_output.get('content_type', 'unknown'),
            'summary': {
                'is_valid': normalized_output['validation'].get('is_valid', False),
                'category': normalized_output['classification'].get('category', 'unknown'),
                'confidence': normalized_output['classification'].get('confidence', 0.0)
            },
            'details': {
                'validation': normalized_output['validation'],
                'classification': normalized_output['classification']
            }
        }
        
        # Add metadata if present
        if normalized_output.get('metadata'):
            dashboard_output['metadata'] = normalized_output['metadata']
        
        # Add errors if present
        if normalized_output.get('errors'):
            dashboard_output['errors'] = normalized_output['errors']
        
        return dashboard_output
    
    def to_json(self, normalized_output: Dict[str, Any], indent: int = 2) -> str:
        """
        Convert normalized output to JSON string.
        
        Args:
            normalized_output: Normalized output
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        try:
            return json.dumps(normalized_output, indent=indent, default=str)
        except Exception as e:
            logger.error(f"Error converting to JSON: {str(e)}")
            return json.dumps(self._create_error_output(str(e)), indent=indent, default=str)
    
    def from_json(self, json_string: str) -> Dict[str, Any]:
        """
        Parse JSON string to normalized output.
        
        Args:
            json_string: JSON string
            
        Returns:
            Normalized output
        """
        try:
            data = json.loads(json_string)
            return self.normalize(data)
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            return self._create_error_output(f"JSON parsing error: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on output normalizer.
        
        Returns:
            Health status information
        """
        return {
            'status': 'healthy',
            'component': 'output_normalizer',
            'standard_format_keys': list(self.standard_format.keys())
        }
    
    def get_standard_format(self) -> Dict[str, Any]:
        """
        Get the standard output format.
        
        Returns:
            Standard format dictionary
        """
        return self.standard_format.copy()
