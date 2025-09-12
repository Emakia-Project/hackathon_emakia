"""
Unit tests for the output normalizer component.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any

from src.pipeline.output_normalizer import OutputNormalizer


class TestOutputNormalizer:
    """Test cases for the OutputNormalizer class."""
    
    @pytest.fixture
    def normalizer(self):
        """Create an output normalizer instance."""
        return OutputNormalizer()
    
    def test_init(self, normalizer):
        """Test normalizer initialization."""
        assert normalizer.standard_format is not None
        assert 'status' in normalizer.standard_format
        assert 'timestamp' in normalizer.standard_format
        assert 'validation' in normalizer.standard_format
        assert 'classification' in normalizer.standard_format
    
    def test_normalize_basic_output(self, normalizer):
        """Test normalizing basic pipeline output."""
        pipeline_output = {
            'validation': {
                'is_valid': True,
                'confidence': 0.9,
                'violations': [],
                'suggestions': ['Good content']
            },
            'classification': {
                'category': 'safe',
                'confidence': 0.85,
                'reasoning': 'Content appears safe',
                'all_categories': ['safe', 'unsafe']
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert result['status'] == 'success'
        assert result['timestamp'] is not None
        assert result['validation']['is_valid'] is True
        assert result['validation']['confidence'] == 0.9
        assert result['classification']['category'] == 'safe'
        assert result['classification']['confidence'] == 0.85
    
    def test_normalize_validation_only(self, normalizer):
        """Test normalizing output with only validation results."""
        pipeline_output = {
            'validation': {
                'is_valid': False,
                'confidence': 0.7,
                'violations': ['Contains inappropriate content'],
                'suggestions': ['Remove inappropriate language']
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert result['status'] == 'validation_failed'
        assert result['validation']['is_valid'] is False
        assert len(result['validation']['violations']) == 1
        assert len(result['validation']['suggestions']) == 1
        assert result['classification']['category'] == 'unknown'
    
    def test_normalize_classification_only(self, normalizer):
        """Test normalizing output with only classification results."""
        pipeline_output = {
            'classification': {
                'category': 'hate_speech',
                'confidence': 0.95,
                'reasoning': 'Contains discriminatory language',
                'all_categories': ['safe', 'hate_speech']
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert result['status'] == 'content_flagged'
        assert result['classification']['category'] == 'hate_speech'
        assert result['classification']['confidence'] == 0.95
        assert result['validation']['is_valid'] is False
    
    def test_normalize_with_errors(self, normalizer):
        """Test normalizing output with errors."""
        pipeline_output = {
            'validation': {
                'is_valid': True,
                'confidence': 0.8
            },
            'errors': ['API rate limit exceeded']
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert result['status'] == 'error'
        assert len(result['errors']) == 1
        assert 'API rate limit exceeded' in result['errors']
    
    def test_normalize_with_metadata(self, normalizer):
        """Test normalizing output with metadata."""
        pipeline_output = {
            'validation': {
                'is_valid': True,
                'confidence': 0.9
            },
            'metadata': {
                'processing_time': 1.5,
                'model_used': 'gpt-4',
                'tokens_used': 200
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert result['metadata']['processing_time'] == 1.5
        assert result['metadata']['model_used'] == 'gpt-4'
        assert result['metadata']['tokens_used'] == 200
    
    def test_normalize_validation_details(self, normalizer):
        """Test normalizing validation with detailed results."""
        pipeline_output = {
            'validation': {
                'is_valid': True,
                'confidence': 0.9,
                'validation_type': 'ai_model',
                'model_provider': 'openai',
                'validation_details': {
                    'basic_validation': {
                        'is_valid': True,
                        'confidence': 1.0
                    },
                    'ai_validation': {
                        'is_valid': True,
                        'confidence': 0.9
                    }
                }
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert result['validation']['validation_type'] == 'ai_model'
        assert result['validation']['model_provider'] == 'openai'
        assert 'basic_validation' in result['validation']['details']
        assert 'ai_validation' in result['validation']['details']
    
    def test_normalize_classification_thresholds(self, normalizer):
        """Test normalizing classification with threshold information."""
        pipeline_output = {
            'classification': {
                'category': 'safe',
                'confidence': 0.75,
                'threshold_met': True,
                'threshold': 0.7,
                'classification_type': 'ai_model',
                'model_provider': 'openai'
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert result['classification']['threshold_met'] is True
        assert result['classification']['threshold'] == 0.7
        assert result['classification']['classification_type'] == 'ai_model'
        assert result['classification']['model_provider'] == 'openai'
    
    def test_normalize_malformed_violations(self, normalizer):
        """Test normalizing with malformed violations (not a list)."""
        pipeline_output = {
            'validation': {
                'is_valid': False,
                'violations': 'Single violation string'
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert isinstance(result['validation']['violations'], list)
        assert len(result['validation']['violations']) == 1
        assert result['validation']['violations'][0] == 'Single violation string'
    
    def test_normalize_malformed_suggestions(self, normalizer):
        """Test normalizing with malformed suggestions (not a list)."""
        pipeline_output = {
            'validation': {
                'is_valid': True,
                'suggestions': 'Single suggestion string'
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert isinstance(result['validation']['suggestions'], list)
        assert len(result['validation']['suggestions']) == 1
        assert result['validation']['suggestions'][0] == 'Single suggestion string'
    
    def test_normalize_malformed_categories(self, normalizer):
        """Test normalizing with malformed categories (not a list)."""
        pipeline_output = {
            'classification': {
                'category': 'safe',
                'all_categories': 'single_category'
            }
        }
        
        result = normalizer.normalize(pipeline_output)
        
        assert isinstance(result['classification']['all_categories'], list)
        assert len(result['classification']['all_categories']) == 1
        assert result['classification']['all_categories'][0] == 'single_category'
    
    def test_determine_overall_status_success(self, normalizer):
        """Test determining overall status for successful validation."""
        normalized_output = {
            'errors': [],
            'validation': {'is_valid': True},
            'classification': {'category': 'safe'}
        }
        
        status = normalizer._determine_overall_status(normalized_output)
        assert status == 'success'
    
    def test_determine_overall_status_error(self, normalizer):
        """Test determining overall status with errors."""
        normalized_output = {
            'errors': ['Some error'],
            'validation': {'is_valid': True},
            'classification': {'category': 'safe'}
        }
        
        status = normalizer._determine_overall_status(normalized_output)
        assert status == 'error'
    
    def test_determine_overall_status_validation_failed(self, normalizer):
        """Test determining overall status when validation fails."""
        normalized_output = {
            'errors': [],
            'validation': {'is_valid': False},
            'classification': {'category': 'safe'}
        }
        
        status = normalizer._determine_overall_status(normalized_output)
        assert status == 'validation_failed'
    
    def test_determine_overall_status_content_flagged(self, normalizer):
        """Test determining overall status when content is flagged."""
        normalized_output = {
            'errors': [],
            'validation': {'is_valid': True},
            'classification': {'category': 'hate_speech'}
        }
        
        status = normalizer._determine_overall_status(normalized_output)
        assert status == 'content_flagged'
    
    def test_create_error_output(self, normalizer):
        """Test creating standardized error output."""
        error_message = "Test error message"
        result = normalizer._create_error_output(error_message)
        
        assert result['status'] == 'error'
        assert result['timestamp'] is not None
        assert error_message in result['errors']
        assert result['validation']['is_valid'] is False
        assert result['classification']['category'] == 'unknown'
    
    def test_format_for_api(self, normalizer):
        """Test formatting output for API response."""
        normalized_output = {
            'status': 'success',
            'timestamp': '2024-01-15T10:30:00Z',
            'validation': {'is_valid': True},
            'classification': {'category': 'safe'},
            'metadata': {'processing_time': 1.5}
        }
        
        api_output = normalizer.format_for_api(normalized_output)
        
        assert api_output['success'] is True
        assert api_output['status'] == 'success'
        assert 'validation' in api_output['data']
        assert 'classification' in api_output['data']
        assert 'metadata' in api_output['data']
    
    def test_format_for_api_with_errors(self, normalizer):
        """Test formatting API output with errors."""
        normalized_output = {
            'status': 'error',
            'timestamp': '2024-01-15T10:30:00Z',
            'validation': {'is_valid': False},
            'classification': {'category': 'unknown'},
            'errors': ['API error']
        }
        
        api_output = normalizer.format_for_api(normalized_output)
        
        assert api_output['success'] is False
        assert 'errors' in api_output
        assert 'API error' in api_output['errors']
    
    def test_format_for_dashboard(self, normalizer):
        """Test formatting output for dashboard display."""
        normalized_output = {
            'status': 'success',
            'timestamp': '2024-01-15T10:30:00Z',
            'content_type': 'text',
            'validation': {'is_valid': True, 'confidence': 0.9},
            'classification': {'category': 'safe', 'confidence': 0.85},
            'metadata': {'processing_time': 1.5}
        }
        
        dashboard_output = normalizer.format_for_dashboard(normalized_output)
        
        assert dashboard_output['status'] == 'success'
        assert dashboard_output['content_type'] == 'text'
        assert 'summary' in dashboard_output
        assert 'details' in dashboard_output
        assert dashboard_output['summary']['is_valid'] is True
        assert dashboard_output['summary']['category'] == 'safe'
        assert dashboard_output['summary']['confidence'] == 0.85
    
    def test_to_json(self, normalizer):
        """Test converting normalized output to JSON."""
        normalized_output = {
            'status': 'success',
            'timestamp': '2024-01-15T10:30:00Z',
            'validation': {'is_valid': True},
            'classification': {'category': 'safe'}
        }
        
        json_string = normalizer.to_json(normalized_output)
        
        # Should be valid JSON
        parsed = json.loads(json_string)
        assert parsed['status'] == 'success'
        assert parsed['validation']['is_valid'] is True
    
    def test_to_json_with_error(self, normalizer):
        """Test JSON conversion with error handling."""
        # Create output with non-serializable object
        normalized_output = {
            'status': 'success',
            'timestamp': datetime.now(),  # This will cause serialization issue
            'validation': {'is_valid': True}
        }
        
        json_string = normalizer.to_json(normalized_output)
        
        # Should handle the error and return valid JSON
        parsed = json.loads(json_string)
        assert 'timestamp' in parsed
    
    def test_from_json(self, normalizer):
        """Test parsing JSON string to normalized output."""
        json_string = '''
        {
            "validation": {
                "is_valid": true,
                "confidence": 0.9
            },
            "classification": {
                "category": "safe",
                "confidence": 0.85
            }
        }
        '''
        
        result = normalizer.from_json(json_string)
        
        assert result['status'] == 'success'
        assert result['validation']['is_valid'] is True
        assert result['classification']['category'] == 'safe'
    
    def test_from_json_invalid(self, normalizer):
        """Test parsing invalid JSON string."""
        invalid_json = "invalid json string"
        
        result = normalizer.from_json(invalid_json)
        
        assert result['status'] == 'error'
        assert 'JSON parsing error' in result['errors'][0]
    
    def test_health_check(self, normalizer):
        """Test health check functionality."""
        health = normalizer.health_check()
        
        assert health['status'] == 'healthy'
        assert health['component'] == 'output_normalizer'
        assert 'standard_format_keys' in health
    
    def test_get_standard_format(self, normalizer):
        """Test getting standard format."""
        format_dict = normalizer.get_standard_format()
        
        assert isinstance(format_dict, dict)
        assert 'status' in format_dict
        assert 'timestamp' in format_dict
        assert 'validation' in format_dict
        assert 'classification' in format_dict
        
        # Should be a copy, not the original
        assert format_dict is not normalizer.standard_format


if __name__ == "__main__":
    pytest.main([__file__])
