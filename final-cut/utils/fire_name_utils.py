#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire Name Utilities - Fire Name Normalization Tools

Provides unified fire name processing logic with support for:
- CSV filename normalization (e.g., "Dolan_gt.csv" -> "DOLAN")
- INCIDENT_NAME field normalization (e.g., "NORTH COMPLEX" -> "NORTH_COMPLEX")
- Automatic matching to config.FIRE_NAMES standard format
"""

from typing import Optional
from config import FIRE_NAMES


def normalize_fire_name(name: str) -> str:
    """
    Normalize fire name to config.FIRE_NAMES standard format.
    
    Handles various input formats including CSV filenames and database field values,
    and normalizes them to match the standard format defined in config.FIRE_NAMES.
    
    Args:
        name: Original fire name, which may come from:
            - CSV filename (e.g., "Dolan_gt.csv", "North_Complex_gt.csv")
            - INCIDENT_NAME field (e.g., "DOLAN", "CREEK", "NORTH COMPLEX")
    
    Returns:
        Normalized fire name (e.g., "DOLAN", "NORTH_COMPLEX")
    
    Examples:
        >>> normalize_fire_name("Dolan_gt.csv")
        'DOLAN'
        >>> normalize_fire_name("NORTH COMPLEX")
        'NORTH_COMPLEX'
        >>> normalize_fire_name("creek")
        'CREEK'
    """
    if not name:
        return name
    
    # Step 1: Remove _gt.csv suffix if present
    cleaned = name.replace('_gt.csv', '').replace('.csv', '')
    
    # Step 2: Convert to uppercase and normalize separators
    cleaned = cleaned.upper().strip()
    
    # Step 3: Try direct match (handles already standard format cases)
    if cleaned in FIRE_NAMES:
        return cleaned
    
    # Step 4: Try matching after converting spaces to underscores
    with_underscores = cleaned.replace(' ', '_').replace('.', '')
    if with_underscores in FIRE_NAMES:
        return with_underscores
    
    # Step 5: Try converting underscores to spaces then back (handles mixed formats)
    # Example: "North_Complex" -> "NORTH_COMPLEX"
    with_spaces = cleaned.replace('_', ' ')
    for standard_name in FIRE_NAMES:
        standard_with_spaces = standard_name.replace('_', ' ')
        if with_spaces == standard_with_spaces:
            return standard_name
    
    # Step 6: If no match found, return underscore version (default format)
    return with_underscores


def match_fire_name_to_standard(name: str) -> Optional[str]:
    """
    Match fire name to standard name in config.FIRE_NAMES.
    
    Returns None if no match is found (for strict validation scenarios).
    This function is more conservative than normalize_fire_name, returning None
    rather than a best-guess normalization when no exact match is found.
    
    Args:
        name: Original fire name
    
    Returns:
        Standard fire name if match found, None otherwise
    """
    normalized = normalize_fire_name(name)
    
    if normalized in FIRE_NAMES:
        return normalized
    
    # Try fuzzy matching by removing underscores
    for standard_name in FIRE_NAMES:
        if normalized.replace('_', '') == standard_name.replace('_', ''):
            return standard_name
    
    return None
