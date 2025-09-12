"""
Unit tests for the validator component.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.pipeline.validator import Validator


class TestValidator:
    """Test cases for the Validator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return {
            'models': {
                'default': 'openai',
                'providers': {
                    'openai': {
                        'api_key': 'test_key',
                        'default_model': 'gpt-4',
                        'max_tokens': 4096,
                        'temperature': 0.1,
                        'timeout': 30
                    }
                }
            },
            'validation': {
                'threshold': 0.8,
                'confidence_threshold': 0.7,
                'max_retries': 3,
                'retry_delay': 1.0,
                'content_types': {
                    'text': {
                        'max_length': 10000,
                        'min_length': 10,
                        'allowed_languages': ['en', 'es', 'fr']
                    }
                }
            }
        }
    
    @pytest.fixture
    def validator(self, mock_config):
        """Create a validator instance with mock configuration."""
        with patch('src.pipeline.validator.OpenAIWrapper') as mock_wrapper:
            mock_wrapper.return_value = Mock()
            return Validator(mock_config)
    
    def test_init(self, mock_config):
        """Test validator initialization."""
        with patch('src.pipeline.validator.OpenAIWrapper') as mock_wrapper:
            mock_wrapper.return_value = Mock()
            validator = Validator(mock_config)
            
            assert validator.config == mock_config
            assert validator.validation_config == mock_config['validation']
            assert 'openai' in validator.model_wrappers
    
    def test_validate_basic_rules_text_valid(self, validator):
        """Test basic rule validation for valid text."""
        content = "This is a valid text content that meets all requirements."
        result = validator._validate_basic_rules(content, "text")
        
        assert result['is_valid'] is True
        assert result['confidence'] == 1.0
        assert result['violations'] == []
        assert result['validation_type'] == 'basic_rules'
    
    def test_validate_basic_rules_text_too_short(self, validator):
        """Test basic rule validation for text that's too short."""
        content = "Short"
        result = validator._validate_basic_rules(content, "text")
        
        assert result['is_valid'] is False
        assert result['confidence'] == 0.0
        assert len(result['violations']) > 0
        assert "too short" in result['violations'][0]
    
    def test_validate_basic_rules_text_too_long(self, validator):
        """Test basic rule validation for text that's too long."""
        content = "x" * 15000  # Exceeds max_length
        result = validator._validate_basic_rules(content, "text")
        
        assert result['is_valid'] is False
        assert result['confidence'] == 0.0
        assert len(result['violations']) > 0
        assert "too long" in result['violations'][0]
    
    def test_validate_basic_rules_empty_content(self, validator):
        """Test basic rule validation for empty content."""
        content = ""
        result = validator._validate_basic_rules(content, "text")
        
        assert result['is_valid'] is False
        assert result['confidence'] == 0.0
        assert len(result['violations']) > 0
        assert "empty" in result['violations'][0]
    
    def test_get_validation_rules_text(self, validator):
        """Test getting validation rules for text content."""
        rules = validator._get_validation_rules("text")
        
        expected_rules = [
            "no_hate_speech",
            "no_violence", 
            "no_harmful_content",
            "appropriate_language",
            "no_spam"
        ]
        
        for rule in expected_rules:
            assert rule in rules
    
    def test_get_validation_rules_image(self, validator):
        """Test getting validation rules for image content."""
        rules = validator._get_validation_rules("image")
        
        # Should include base rules plus image-specific rules
        assert "no_hate_speech" in rules
        assert "no_explicit_content" in rules
        assert "no_violence" in rules
    
    def test_get_validation_rules_video(self, validator):
        """Test getting validation rules for video content."""
        rules = validator._get_validation_rules("video")
        
        # Should include base rules plus video-specific rules
        assert "no_hate_speech" in rules
        assert "no_explicit_content" in rules
        assert "appropriate_duration" in rules
    
    def test_combine_validation_results(self, validator):
        """Test combining validation results."""
        basic_result = {
            'is_valid': True,
            'violations': [],
            'suggestions': ['Good content'],
            'confidence': 1.0,
            'validation_type': 'basic_rules'
        }
        
        ai_result = {
            'is_valid': True,
            'violations': [],
            'suggestions': ['Well written'],
            'confidence': 0.9,
            'validation_type': 'ai_model',
            'model_provider': 'openai'
        }
        
        combined = validator._combine_validation_results(basic_result, ai_result)
        
        assert combined['is_valid'] is True
        assert len(combined['violations']) == 0
        assert len(combined['suggestions']) == 2
        assert combined['confidence'] == pytest.approx(0.93, rel=1e-2)  # Weighted average
    
    def test_combine_validation_results_with_violations(self, validator):
        """Test combining validation results with violations."""
        basic_result = {
            'is_valid': True,
            'violations': [],
            'suggestions': [],
            'confidence': 1.0,
            'validation_type': 'basic_rules'
        }
        
        ai_result = {
            'is_valid': False,
            'violations': ['Contains inappropriate content'],
            'suggestions': ['Remove inappropriate language'],
            'confidence': 0.8,
            'validation_type': 'ai_model',
            'model_provider': 'openai'
        }
        
        combined = validator._combine_validation_results(basic_result, ai_result)
        
        assert combined['is_valid'] is False
        assert len(combined['violations']) == 1
        assert len(combined['suggestions']) == 1
        assert combined['confidence'] == pytest.approx(0.86, rel=1e-2)
    
    @pytest.mark.asyncio
    async def test_validate_with_ai_success(self, validator):
        """Test AI validation with successful result."""
        # Mock the wrapper
        mock_wrapper = AsyncMock()
        mock_wrapper.validate.return_value = {
            'is_valid': True,
            'violations': [],
            'suggestions': [],
            'confidence': 0.9,
            'model_provider': 'openai'
        }
        validator.model_wrappers['openai'] = mock_wrapper
        
        result = await validator._validate_with_ai("Test content", "text")
        
        assert result['is_valid'] is True
        assert result['validation_type'] == 'ai_model'
        assert result['model_provider'] == 'openai'
        mock_wrapper.validate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_with_ai_fallback(self, validator):
        """Test AI validation with fallback when default provider unavailable."""
        # Remove default provider
        validator.model_wrappers.pop('openai', None)
        
        # Add fallback provider
        mock_wrapper = AsyncMock()
        mock_wrapper.validate.return_value = {
            'is_valid': True,
            'violations': [],
            'suggestions': [],
            'confidence': 0.8,
            'model_provider': 'fireworks'
        }
        validator.model_wrappers['fireworks'] = mock_wrapper
        
        result = await validator._validate_with_ai("Test content", "text")
        
        assert result['is_valid'] is True
        assert result['model_provider'] == 'fireworks'
    
    @pytest.mark.asyncio
    async def test_validate_with_ai_no_providers(self, validator):
        """Test AI validation when no providers are available."""
        validator.model_wrappers.clear()
        
        result = await validator._validate_with_ai("Test content", "text")
        
        assert result['is_valid'] is False
        assert result['validation_type'] == 'ai_fallback'
        assert 'No AI models available' in result['violations'][0]
    
    @pytest.mark.asyncio
    async def test_validate_success(self, validator):
        """Test complete validation process."""
        # Mock basic validation
        with patch.object(validator, '_validate_basic_rules') as mock_basic:
            mock_basic.return_value = {
                'is_valid': True,
                'violations': [],
                'suggestions': [],
                'confidence': 1.0,
                'validation_type': 'basic_rules'
            }
            
            # Mock AI validation
            with patch.object(validator, '_validate_with_ai') as mock_ai:
                mock_ai.return_value = {
                    'is_valid': True,
                    'violations': [],
                    'suggestions': [],
                    'confidence': 0.9,
                    'validation_type': 'ai_model',
                    'model_provider': 'openai'
                }
                
                result = await validator.validate("Test content", "text")
                
                assert result['is_valid'] is True
                assert result['status'] == 'success'
                assert 'validation' in result
                assert 'classification' in result
    
    @pytest.mark.asyncio
    async def test_validate_error_handling(self, validator):
        """Test validation error handling."""
        with patch.object(validator, '_validate_basic_rules', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                await validator.validate("Test content", "text")
    
    def test_health_check(self, validator):
        """Test health check functionality."""
        health = validator.health_check()
        
        assert health['status'] == 'healthy'
        assert health['component'] == 'validator'
        assert 'model_wrappers' in health
    
    def test_health_check_no_wrappers(self, validator):
        """Test health check when no model wrappers are available."""
        validator.model_wrappers.clear()
        
        health = validator.health_check()
        
        assert health['status'] == 'warning'
        assert 'No model wrappers available' in health['message']
    
    @pytest.mark.asyncio
    async def test_validate_batch(self, validator):
        """Test batch validation."""
        contents = ["Content 1", "Content 2", "Content 3"]
        
        # Mock the validate method
        with patch.object(validator, 'validate') as mock_validate:
            mock_validate.return_value = {
                'is_valid': True,
                'status': 'success'
            }
            
            results = await validator.validate_batch(contents, "text")
            
            assert len(results) == 3
            assert mock_validate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_validate_batch_with_errors(self, validator):
        """Test batch validation with some errors."""
        contents = ["Content 1", "Content 2", "Content 3"]
        
        # Mock the validate method to raise an exception for one item
        with patch.object(validator, 'validate') as mock_validate:
            mock_validate.side_effect = [
                {'is_valid': True, 'status': 'success'},
                Exception("Test error"),
                {'is_valid': True, 'status': 'success'}
            ]
            
            results = await validator.validate_batch(contents, "text")
            
            # Should have 2 successful results and 1 error logged
            assert len(results) == 2
            assert mock_validate.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
