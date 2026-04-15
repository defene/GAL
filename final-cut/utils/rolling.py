#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling Metrics Utilities - Shared Module for Rolling Metric Calculations

Extracted rolling calculation and formatting functions from overall_pipeline.py,
providing unified implementation for both overall_pipeline and generate_trend_features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from config import ROLLING_CONFIG


def compute_rolling_metrics(history_buffer: List[Dict[str, float]], current_values: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate rolling metrics for fire data.
    
    Computes rolling averages, maximums, ratios to historical max, and time-based
    features for fire points and area metrics.
    
    Args:
        history_buffer: Historical values (list of dictionaries)
        current_values: Current day values (dictionary)
        
    Returns:
        Dictionary containing rolling metrics for all configured variables
    """
    # Add current values to history buffer
    all_values = history_buffer + [current_values]
    
    # Convert to DataFrame for easier calculations
    df = pd.DataFrame(all_values)
    
    rolling_metrics = {}
    
    # Base variables to compute rolling metrics for
    variables = ['fire_points', 'area_total', 'area_max']
    
    for var in variables:
        if var not in df.columns:
            continue
            
        series = df[var]
        
        # Rolling window calculations
        for window in ROLLING_CONFIG['windows']:
            min_periods = ROLLING_CONFIG['min_periods']
            
            # Mean and maximum
            rolling_mean = series.rolling(window=window, min_periods=min_periods).mean().iloc[-1]
            rolling_max = series.rolling(window=window, min_periods=min_periods).max().iloc[-1]
            
            rolling_metrics[f'rolling{window}_mean_{var}'] = rolling_mean if pd.notna(rolling_mean) else None
            rolling_metrics[f'rolling{window}_max_{var}'] = rolling_max if pd.notna(rolling_max) else None
        
        
        # Ratio to historical maximum and time information
        if ROLLING_CONFIG['ratio_mode'] == 'to_date_max':
            current_val = current_values.get(var)
            if current_val is not None and len(all_values) > 0:
                historical_max = series.max()
                if pd.notna(historical_max) and historical_max > 0:
                    ratio = current_val / historical_max
                    # Find position of maximum value (last occurrence)
                    max_idx = series.idxmax()
                    days_since_max = len(all_values) - 1 - max_idx  # Days from current
                else:
                    ratio = None if len(all_values) > 1 else None
                    historical_max = None
                    days_since_max = None
            else:
                ratio = None
                historical_max = None
                days_since_max = None
            
            rolling_metrics[f'todate_ratio_{var}'] = ratio
            rolling_metrics[f'todate_max_{var}'] = historical_max
            rolling_metrics[f'days_since_max_{var}'] = days_since_max
    
    return rolling_metrics


def format_rolling_for_llm(rolling_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Format rolling metrics for LLM narrative use.
    
    Converts raw rolling metrics to human-readable strings with proper
    formatting for numerical values and time expressions.
    
    Args:
        rolling_metrics: Dictionary of raw rolling metric values
        
    Returns:
        Dictionary with formatted metrics for fire_points and area
    """
    formatted = {}
    
    # Format numerical values, handle None values
    def format_value(val, decimal_places=2):
        if val is None or pd.isna(val):
            return "N/A"
        return f"{val:.{decimal_places}f}"
    
    def format_days(val):
        if val is None or pd.isna(val):
            return "N/A"
        days = int(val)
        if days == 0:
            return "today"
        elif days == 1:
            return "1 day ago"
        else:
            return f"{days} days ago"
    
    # Fire points related metrics
    formatted["fire_points"] = {
        "3day_avg_total": format_value(rolling_metrics.get('rolling3_mean_fire_points')),
        "7day_avg_total": format_value(rolling_metrics.get('rolling7_mean_fire_points')),
        "3day_max_total": format_value(rolling_metrics.get('rolling3_max_fire_points')),
        "7day_max_total": format_value(rolling_metrics.get('rolling7_max_fire_points')),
        "current_vs_max_ratio": format_value(rolling_metrics.get('todate_ratio_fire_points'), 3),
        "global_max_value": format_value(rolling_metrics.get('todate_max_fire_points')),
        "days_since_global_max": format_days(rolling_metrics.get('days_since_max_fire_points'))
    }
    
    # Area related metrics
    formatted["area"] = {
        "3day_avg_total": format_value(rolling_metrics.get('rolling3_mean_area_total')),
        "7day_avg_total": format_value(rolling_metrics.get('rolling7_mean_area_total')),
        "3day_max_total": format_value(rolling_metrics.get('rolling3_max_area_total')),
        "7day_max_total": format_value(rolling_metrics.get('rolling7_max_area_total')),
        "current_vs_max_ratio": format_value(rolling_metrics.get('todate_ratio_area_total'), 3),
        "global_max_value": format_value(rolling_metrics.get('todate_max_area_total')),
        "days_since_global_max": format_days(rolling_metrics.get('days_since_max_area_total'))
    }
    
    return formatted
