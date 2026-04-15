"""
Unit metadata definitions for data sources.

This module defines the original units for all data sources in the system,
providing a central reference for unit conversion in the presentation layer.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class UnitInfo:
    """Information about a data field's unit"""
    unit: str
    description: str
    source: str


# Original unit definitions for all data sources
ORIGINAL_UNITS = {
    # Distance measurements
    "distance_m": UnitInfo("m", "Distance in meters", "PostGIS ST_Distance"),
    "eps_meters": UnitInfo("m", "Clustering radius in meters", "Config parameter"),
    
    # Area measurements  
    "area_m2": UnitInfo("m²", "Area in square meters", "PostGIS ST_Area with EPSG:3310"),
    "cluster_area_m2": UnitInfo("m²", "Cluster area in square meters", "PostGIS ST_Area with EPSG:3310"),
    
    # Temperature measurements
    "brightness": UnitInfo("K", "Brightness temperature in Kelvin (fixed unit)", "Fire point data"),
    "max_brightness": UnitInfo("K", "Maximum brightness temperature in Kelvin (fixed unit)", "Fire point data"),
    "tmmx": UnitInfo("K", "Maximum temperature in Kelvin", "Weather raster data"),
    "tmmn": UnitInfo("K", "Minimum temperature in Kelvin", "Weather raster data"),
    
    # Speed measurements
    "vs": UnitInfo("m/s", "Wind speed in meters per second", "Weather raster data"),
    "wind_speed": UnitInfo("m/s", "Wind speed in meters per second", "Weather raster data"),
    
    # Dimensionless measurements
    "frp": UnitInfo("MW", "Fire radiative power in megawatts", "Fire point data"),
    "bi": UnitInfo("index", "Burning Index (dimensionless)", "Weather raster data"),
    "fm1": UnitInfo("%", "1-hour fuel moisture in percentage", "Weather raster data"),
    "population": UnitInfo("count", "Population count", "Population raster data"),
    "pop_sum": UnitInfo("count", "Population sum", "Population raster data"),
    
    # Coordinate measurements
    "latitude": UnitInfo("degrees", "Latitude in decimal degrees", "WGS84"),
    "longitude": UnitInfo("degrees", "Longitude in decimal degrees", "WGS84"),
    "center_lat": UnitInfo("degrees", "Center latitude in decimal degrees", "WGS84"),
    "center_lon": UnitInfo("degrees", "Center longitude in decimal degrees", "WGS84"),
}


def get_unit_info(field_name: str) -> UnitInfo:
    """Get unit information for a data field"""
    return ORIGINAL_UNITS.get(field_name, UnitInfo("unknown", "Unknown unit", "Unknown source"))


def get_original_unit(field_name: str) -> str:
    """Get the original unit for a data field"""
    return get_unit_info(field_name).unit


def is_distance_field(field_name: str) -> bool:
    """Check if a field represents distance measurement"""
    unit_info = get_unit_info(field_name)
    return unit_info.unit == "m"


def is_area_field(field_name: str) -> bool:
    """Check if a field represents area measurement"""
    unit_info = get_unit_info(field_name)
    return unit_info.unit == "m²"


def is_temperature_field(field_name: str) -> bool:
    """Check if a field represents temperature measurement"""
    unit_info = get_unit_info(field_name)
    return unit_info.unit == "K"


def is_speed_field(field_name: str) -> bool:
    """Check if a field represents speed measurement"""
    unit_info = get_unit_info(field_name)
    return unit_info.unit == "m/s"


def get_field_summary() -> Dict[str, Dict[str, Any]]:
    """Get a summary of all field units organized by category"""
    summary = {
        "distance": {},
        "area": {},
        "temperature": {},
        "speed": {},
        "other": {}
    }
    
    for field_name, unit_info in ORIGINAL_UNITS.items():
        if unit_info.unit == "m":
            summary["distance"][field_name] = unit_info
        elif unit_info.unit == "m²":
            summary["area"][field_name] = unit_info
        elif unit_info.unit == "K":
            summary["temperature"][field_name] = unit_info
        elif unit_info.unit == "m/s":
            summary["speed"][field_name] = unit_info
        else:
            summary["other"][field_name] = unit_info
    
    return summary


def validate_field_units(data: Dict[str, Any]) -> Dict[str, str]:
    """Validate that data fields have expected units (for debugging)"""
    issues = {}
    
    for field_name, value in data.items():
        if field_name in ORIGINAL_UNITS:
            unit_info = ORIGINAL_UNITS[field_name]
            # Basic validation - could be extended
            if value is not None and not isinstance(value, (int, float, str)):
                issues[field_name] = f"Expected numeric value for {unit_info.unit} field"
    
    return issues
