#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Format Utilities - Eliminate duplicate logic, provide concise range formatting

Provides formatting utilities for RAG context strings with support for both
detailed and range-based display formats.
"""

from typing import List, Dict, Any, Tuple, Optional


def format_rag_range_context(results: List[Dict[str, Any]], display_format: str = None) -> str:
    """
    Format RAG context with support for multiple display formats.
    
    Provides two formatting options: 'range' for concise min-max ranges,
    or 'detailed' for individual entry listings.
    
    Args:
        results: List of retrieval results
        display_format: Display format - "range" (concise ranges) or "detailed" (detailed entries),
                       None to read from config
        
    Returns:
        Formatted RAG context string
    """
    if not results:
        return ""
    
    # Get display format from config if not specified
    if display_format is None:
        from config import RAG_CONFIG
        display_format = RAG_CONFIG.get("display_format", "range")
    
    if display_format == "detailed":
        return _format_detailed_context(results)
    else:  # default to "range"
        return _format_range_context(results)


def _format_range_context(results: List[Dict[str, Any]]) -> str:
    """
    Format as concise range format.
    
    Extracts personnel and budget values from results, trims extremes,
    and formats as min-max ranges for bounding reference.
    
    Args:
        results: List of retrieval results
        
    Returns:
        Formatted string with personnel and budget ranges
    """
    # Extract valid Personnel and Daily_Budget values
    personnel_values = []
    budget_values = []
    
    for result in results:
        # Extract Personnel
        personnel = result.get('TOTAL_PERSONNEL', 'N/A')
        if personnel != 'N/A' and personnel is not None:
            try:
                personnel_values.append(float(personnel))
            except (ValueError, TypeError):
                pass
        
        # Extract Daily_Budget
        daily_cost = result.get('EST_IM_COST_TO_DATE_FIXED_DAILY', 'N/A')
        if daily_cost != 'N/A' and daily_cost is not None:
            try:
                budget_values.append(float(daily_cost))
            except (ValueError, TypeError):
                pass
    
    # Generate range format
    lines = ["## RAG References (for bounding)"]
    
    if personnel_values:
        # Trim one maximum and one minimum value
        trimmed_personnel = _trim_extremes(personnel_values)
        if trimmed_personnel:
            min_personnel = min(trimmed_personnel)
            max_personnel = max(trimmed_personnel)
            lines.append(f"- Personnel range: ~{int(min_personnel)}–{int(max_personnel)}")
    
    if budget_values:
        # Trim one maximum and one minimum value
        trimmed_budget = _trim_extremes(budget_values)
        if trimmed_budget:
            min_budget = min(trimmed_budget)
            max_budget = max(trimmed_budget)
            min_budget_str = f"{int(min_budget)}"
            max_budget_str = f"{int(max_budget)}"
            lines.append(f"- Daily budget range: ~${min_budget_str}–${max_budget_str}")
    
    return "\n".join(lines)


def _format_detailed_context(results: List[Dict[str, Any]]) -> str:
    """
    Format as detailed entry format.
    
    Lists each retrieval result with fire name, date, similarity score,
    and ground truth metrics (personnel, daily budget, total budget).
    
    Args:
        results: List of retrieval results
        
    Returns:
        Formatted string with detailed entries
    """
    if not results:
        return ""
    
    lines = []
    for result in results:
        fire_name = result.get('fire_name', 'Unknown')
        mmdd = result.get('mmdd', 'Unknown')
        similarity = float(result.get('similarity', 0.0) or 0.0)
        
        # GT three columns
        personnel = result.get('TOTAL_PERSONNEL', 'N/A')
        daily_cost = result.get('EST_IM_COST_TO_DATE_FIXED_DAILY', 'N/A')
        total_cost = result.get('EST_IM_COST_TO_DATE_FIXED', 'N/A')
        
        line = f"- [{fire_name} {mmdd}] sim={similarity:.3f} | Personnel={personnel}, Daily_Budget=${daily_cost}, Total_Budget=${total_cost}"
        lines.append(line)
    
    return "\n".join(lines)


def _trim_extremes(values: List[float]) -> List[float]:
    """
    Trim one maximum and one minimum value to make range more robust.
    
    Removes outliers by eliminating the highest and lowest values,
    providing a more stable range estimate.
    
    Args:
        values: List of numeric values
        
    Returns:
        List with extreme values removed (empty if insufficient values)
    """
    if len(values) <= 2:
        # If 2 or fewer values, no trimming performed
        return values
    
    # Sort and trim first and last values
    sorted_values = sorted(values)
    return sorted_values[1:-1]
