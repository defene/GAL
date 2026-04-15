#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison Utilities - Unified comparison and formatting logic

Provides standardized functions for formatting value changes, arrows,
and building comparison sections, eliminating duplicate logic across modules.
"""

from typing import Any, Dict, List, Union, Optional


def format_value_with_arrow(
    prev_value: Union[int, float],
    curr_value: Union[int, float],
    is_int: bool = False,
    percent_threshold: float = 0.05,
    precision: int = 1,
    pct_precision: int = 1,
    show_percent: bool = True,
    use_arrow: bool = True
) -> Dict[str, str]:
    """
    Format a value change with arrow and delta.
    
    Unified function replacing: format_arrow, format_delta, arrow_for_int,
    arrow_for_float, fmt_delta_int, fmt_delta_float.
    
    Args:
        prev_value: Previous value
        curr_value: Current value
        is_int: Whether values are integers
        percent_threshold: Minimum percentage change to show arrow (for floats)
        precision: Decimal precision for delta
        pct_precision: Decimal precision for percentage
        show_percent: Whether to include percentage change
        use_arrow: Whether to include directional arrow
        
    Returns:
        Dictionary with 'arrow' and 'delta' keys
    """
    delta = curr_value - prev_value
    
    # Calculate arrow
    arrow = ""
    if use_arrow:
        if is_int:
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        else:
            if prev_value == 0:
                arrow = "↑" if curr_value > 0 else "→"
            else:
                pct_change = abs(delta / prev_value)
                if pct_change >= percent_threshold:
                    arrow = "↑" if delta > 0 else "↓"
                else:
                    arrow = "→"
    
    # Calculate delta string
    if is_int:
        if delta == 0:
            delta_str = "0"
        elif delta > 0:
            delta_str = f"+{int(delta)}"
        else:
            delta_str = str(int(delta))
    else:
        if not show_percent:
            delta_str = f"+{delta:.{precision}f}" if delta >= 0 else f"{delta:.{precision}f}"
        else:
            if prev_value == 0:
                if curr_value == 0:
                    delta_str = "0, 0.0%"
                else:
                    delta_str = f"+{curr_value:.{precision}f}, +∞%"
            else:
                pct_change = (delta / prev_value) * 100
                delta_val = f"+{delta:.{precision}f}" if delta >= 0 else f"{delta:.{precision}f}"
                pct_val = f"+{pct_change:.{pct_precision}f}%" if pct_change >= 0 else f"{pct_change:.{pct_precision}f}%"
                delta_str = f"{delta_val}, {pct_val}"
    
    return {"arrow": arrow, "delta": delta_str}


def build_metric_comparison_line(
    metric_name: str,
    prev_value: Union[int, float],
    curr_value: Union[int, float],
    is_int: bool = False,
    unit: str = "",
    format_func: Optional[callable] = None,
    **kwargs
) -> str:
    """
    Build a single comparison line for a metric.
    
    Args:
        metric_name: Display name for the metric
        prev_value: Previous value
        curr_value: Current value
        is_int: Whether value is integer
        unit: Unit suffix (e.g., "MW", "K")
        format_func: Optional custom formatting function for current value
        **kwargs: Additional arguments for format_value_with_arrow
        
    Returns:
        Formatted comparison line string
    """
    comparison = format_value_with_arrow(prev_value, curr_value, is_int=is_int, **kwargs)
    
    if format_func:
        curr_display = format_func(curr_value)
    else:
        if is_int:
            curr_display = str(int(curr_value))
        else:
            precision = kwargs.get('precision', 1)
            curr_display = f"{curr_value:.{precision}f}"
    
    if unit:
        curr_display = f"{curr_display} {unit}"
    
    return f"- {metric_name}: {curr_display} ({comparison['arrow']} {comparison['delta']})"


def compare_lists(prev_list: List[Any], curr_list: List[Any]) -> Dict[str, Any]:
    """
    Compare two lists and return added/removed items.
    
    Generic version for comparing counties, stations, etc.
    
    Args:
        prev_list: Previous list
        curr_list: Current list
        
    Returns:
        Dictionary with 'added', 'removed', 'unchanged', 'total_now' keys
    """
    prev_set = set(prev_list) if prev_list else set()
    curr_set = set(curr_list) if curr_list else set()
    
    return {
        "added": curr_set - prev_set,
        "removed": prev_set - curr_set,
        "unchanged": len(prev_set & curr_set),
        "total_now": len(curr_set)
    }
