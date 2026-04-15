#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry Utilities - Unified retry logic for API calls

Provides reusable retry decorators and error handling utilities.
"""

import time
from typing import Callable, Any, Optional
from functools import wraps


def is_retryable_error(error_msg: str) -> bool:
    """
    Determine if an error is retryable.
    
    Checks error message against known retryable error patterns like
    rate limits, timeouts, and temporary server errors.
    
    Args:
        error_msg: Error message string
    
    Returns:
        True if error is retryable, False otherwise
    """
    retryable_errors = [
        "internal_error",
        "Internal server error",
        "rate_limit_exceeded",
        "server_error",
        "timeout",
        "connection",
        "temporary"
    ]
    
    error_lower = error_msg.lower()
    return any(retryable_error in error_lower for retryable_error in retryable_errors)


def retry_with_backoff(
    func: Callable,
    max_retries: int,
    model_name: str,
    *args,
    **kwargs
) -> Optional[Any]:
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        model_name: Model name (for logging)
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Function result if successful, None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error calling {model_name} (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
            
            # Check if error is retryable
            if is_retryable_error(error_msg) and attempt < max_retries:
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 4, 8 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Failed after {attempt + 1} attempts. Error: {error_msg}")
                return None
    
    return None


def with_retry(max_retries: int = 3):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        max_retries: Maximum number of retry attempts
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @with_retry(max_retries=3)
        def call_api():
            # API call logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    error_msg = str(e)
                    func_name = func.__name__
                    print(f"Error in {func_name} (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                    
                    if is_retryable_error(error_msg) and attempt < max_retries:
                        wait_time = (2 ** attempt) + 1
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Failed after {attempt + 1} attempts.")
                        return None
            
            return None
        
        return wrapper
    return decorator
