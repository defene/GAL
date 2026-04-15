#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result Builders - Unified result construction utilities

Provides standardized builders for creating consistent result dictionaries
across the codebase, reducing boilerplate code for error/success responses.
"""

from typing import Dict, Any, Optional


def build_result_dict(
    success: bool,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    note: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a standardized result dictionary.
    
    Args:
        success: Whether the operation succeeded
        data: Success data dictionary
        error: Error message if failed
        note: Optional note/warning message
        **kwargs: Additional fields to include
        
    Returns:
        Standardized result dictionary
    """
    result = {}
    
    if data:
        result.update(data)
    
    if error:
        result['error'] = error
    
    if note:
        result['note'] = note
    
    result.update(kwargs)
    
    return result


def build_error_result(
    error_msg: str,
    base_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a standardized error result.
    
    Args:
        error_msg: Error message
        base_data: Base data to include in result
        **kwargs: Additional fields
        
    Returns:
        Error result dictionary
    """
    result = base_data.copy() if base_data else {}
    result['error'] = error_msg
    result.update(kwargs)
    return result


def build_success_result(
    data: Dict[str, Any],
    note: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a standardized success result.
    
    Args:
        data: Success data
        note: Optional note
        **kwargs: Additional fields
        
    Returns:
        Success result dictionary
    """
    result = data.copy()
    if note:
        result['note'] = note
    result.update(kwargs)
    return result
