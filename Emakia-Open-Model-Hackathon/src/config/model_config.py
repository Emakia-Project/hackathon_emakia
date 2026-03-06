"""
Model configuration management for the Emakia Validator Agent.

This module handles loading configuration from YAML files and environment variables,
providing a centralized configuration management system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Global configuration cache
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to configuration file. If None, uses default path.
        
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    # Load environment variables
    load_dotenv()
    
    # Determine config file path
    if config_path is None:
        config_path = Path(__file__).parent / "model_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variable placeholders
    config = _replace_env_vars(config)
    
    # Validate configuration
    _validate_config(config)
    
    # Cache configuration
    _config_cache = config
    
    return config


def _replace_env_vars(config: Any) -> Any:
    """
    Recursively replace environment variable placeholders in configuration.
    
    Args:
        config: Configuration object (dict, list, or primitive)
        
    Returns:
        Configuration with environment variables replaced
    """
    if isinstance(config, dict):
        return {key: _replace_env_vars(value) for key, value in config.items()}
    
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]  # Remove ${ and }
        return os.getenv(env_var, "")
    
    else:
        return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['models', 'validation', 'classification', 'pipeline']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model providers
    if 'providers' not in config['models']:
        raise ValueError("Missing 'providers' in models configuration")
    
    # Validate validation settings
    if 'threshold' not in config['validation']:
        raise ValueError("Missing 'threshold' in validation configuration")
    
    # Validate classification settings
    if 'categories' not in config['classification']:
        raise ValueError("Missing 'categories' in classification configuration")


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        Configuration dictionary
    """
    if _config_cache is None:
        return load_config()
    return _config_cache


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of configuration updates
    """
    global _config_cache
    
    if _config_cache is None:
        _config_cache = load_config()
    
    _update_nested_dict(_config_cache, updates)


def _update_nested_dict(base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """
    Recursively update nested dictionary.
    
    Args:
        base_dict: Base dictionary to update
        updates: Updates to apply
    """
    for key, value in updates.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _update_nested_dict(base_dict[key], value)
        else:
            base_dict[key] = value


def get_model_config(provider: str = None) -> Dict[str, Any]:
    """
    Get configuration for a specific model provider.
    
    Args:
        provider: Model provider name. If None, returns default provider.
        
    Returns:
        Model provider configuration
    """
    config = get_config()
    
    if provider is None:
        provider = config['models']['default']
    
    if provider not in config['models']['providers']:
        raise ValueError(f"Unknown model provider: {provider}")
    
    return config['models']['providers'][provider]


def get_validation_config() -> Dict[str, Any]:
    """
    Get validation configuration.
    
    Returns:
        Validation configuration
    """
    return get_config()['validation']


def get_classification_config() -> Dict[str, Any]:
    """
    Get classification configuration.
    
    Returns:
        Classification configuration
    """
    return get_config()['classification']


def get_pipeline_config() -> Dict[str, Any]:
    """
    Get pipeline configuration.
    
    Returns:
        Pipeline configuration
    """
    return get_config()['pipeline']


def reset_config() -> None:
    """
    Reset configuration cache to force reload.
    """
    global _config_cache
    _config_cache = None
