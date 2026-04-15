#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Utilities - Shared Date Processing Functions

Provides common date manipulation and feature extraction functions used across
the RAG and analysis modules.
"""

import math
from typing import List, Optional
from datetime import datetime

from config import RAG_CONFIG


def get_day_of_year(month: int, day: int) -> int:
    """
    Calculate day of year from month and day (ignore year).
    
    Uses leap year 2020 as reference to handle February 29.
    
    Args:
        month: Month (1-12)
        day: Day of month (1-31)
        
    Returns:
        Day of year (1-366), defaults to 1 on error
    """
    try:
        date_obj = datetime(2020, month, day)  # Use leap year 2020
        return date_obj.timetuple().tm_yday
    except (ValueError, TypeError):
        return 1  # Default to day 1


def extract_date_cyclic_features(mmdd: str, cycle_days: int = None) -> List[float]:
    """
    Extract date cyclic features using sine/cosine encoding.
    
    Converts MM-DD date string to two-dimensional cyclic features that capture
    seasonal periodicity. Returns [sin, cos] encoding of day of year.
    
    Args:
        mmdd: Date string in MM-DD format (e.g., "08-20")
        cycle_days: Days in cycle (defaults to RAG_CONFIG value or 366)
        
    Returns:
        List of two floats [date_sin, date_cos], defaults to [0.0, 1.0] for January 1
    """
    if not mmdd or len(mmdd) < 5:  # MM-DD requires at least 5 characters
        return [0.0, 1.0]
    
    try:
        month, day = map(int, mmdd.split("-"))
        doy = get_day_of_year(month, day)
        
        if cycle_days is None:
            cycle_days = RAG_CONFIG.get("date_cycle_days", 366)
        
        date_sin = math.sin(2 * math.pi * doy / cycle_days)
        date_cos = math.cos(2 * math.pi * doy / cycle_days)
        
        return [date_sin, date_cos]
    except (ValueError, TypeError, AttributeError):
        return [0.0, 1.0]  # Default value corresponds to January 1


def extract_mmdd_from_date(date_str: str) -> Optional[str]:
    """
    Extract MM-DD format from date string.
    
    Supports YYYY-MM-DD format and extracts the month-day portion.
    
    Args:
        date_str: Date string (e.g., "2020-08-20")
        
    Returns:
        MM-DD format string (e.g., "08-20"), or None on failure
    """
    try:
        if not date_str or len(date_str) < 10:
            return None
        # Assume date format is YYYY-MM-DD
        return date_str[5:10]  # Extract MM-DD
    except (IndexError, TypeError, AttributeError):
        return None


def parse_mmdd_string(mmdd: str) -> Optional[tuple]:
    """
    Parse MM-DD string into (month, day) tuple.
    
    Args:
        mmdd: Date string in MM-DD format
        
    Returns:
        (month, day) tuple, or None on failure
    """
    try:
        if not mmdd or len(mmdd) < 5:
            return None
        month, day = map(int, mmdd.split("-"))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (month, day)
        return None
    except (ValueError, TypeError, AttributeError):
        return None
