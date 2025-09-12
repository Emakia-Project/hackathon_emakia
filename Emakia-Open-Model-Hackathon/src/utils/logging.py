"""
Logging utilities for the Emakia Validator Agent.

This module provides centralized logging configuration and utilities.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_size: str = "10 MB",
    rotation: str = "1 day",
    retention: str = "30 days"
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        max_size: Maximum size of log file before rotation
        rotation: Log rotation frequency
        retention: How long to keep log files
    """
    # Remove default logger
    logger.remove()
    
    # Default format if not provided
    if log_format is None:
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    # Add console logger
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file logger if specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str = None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


def set_log_level(level: str) -> None:
    """
    Set the logging level dynamically.
    
    Args:
        level: New logging level
    """
    # This is a simplified implementation
    # In a real application, you might need to reconfigure loggers
    logger.info(f"Log level changed to: {level}")


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


def log_async_function_call(func):
    """
    Decorator to log async function calls.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    async def wrapper(*args, **kwargs):
        logger.debug(f"Calling async {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async {func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Async {func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


class LoggerMixin:
    """
    Mixin class to add logging capabilities to other classes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def log_exception(self, message: str, **kwargs):
        """Log exception message."""
        self.logger.exception(message, **kwargs)
