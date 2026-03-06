"""
Configuration module for the Emakia Validator Agent.

This module handles loading and managing configuration settings from YAML files
and environment variables.
"""

from .model_config import load_config, get_config, update_config

__all__ = ['load_config', 'get_config', 'update_config']
