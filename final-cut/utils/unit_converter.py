"""
Unified unit conversion module supporting multiple unit systems for A/B/C experiments.

This module provides centralized unit conversion and formatting capabilities
to replace scattered conversion constants throughout the codebase.
"""

from typing import Dict, Any, Union, Optional
from enum import Enum


class UnitSystem(Enum):
    """Unit system definitions for A/B/C experiments"""
    A = "A"  # Small values: km², °C, m/s, mile
    B = "B"  # Standard: acres, ℉, m/s, mile  
    C = "C"  # Large values: m², K, km/h, km


# Conversion factors to base units (meters, square meters, Kelvin, m/s)
CONVERSION_FACTORS = {
    # Area conversions (to/from square meters)
    'area': {
        'm2': 1.0,
        'km2': 1e-6,
        'acres': 0.000247105,
        'ha': 1e-4,  # hectares
    },
    
    # Distance conversions (to/from meters) 
    'distance': {
        'm': 1.0,
        'km': 0.001,
        'mile': 0.000621371,
        'ft': 3.28084,
    },
    
    # Temperature conversions (to/from Kelvin)
    'temperature': {
        'K': 1.0,
        'C': lambda k: k - 273.15,
        'F': lambda k: (k - 273.15) * 9/5 + 32,
    },
    
    # Speed conversions (to/from m/s)
    'speed': {
        'ms': 1.0,  # m/s
        'kmh': 3.6,  # km/h
        'mph': 2.23694,  # miles per hour
        'kts': 1.94384,  # knots
    }
}

# Unit system definitions
UNIT_SYSTEMS = {
    UnitSystem.A: {
        'area': 'km2',
        'distance': 'mile', 
        'temperature': 'C',
        'speed': 'ms',
        'labels': {
            'area': 'km²',
            'distance': 'mile',
            'temperature': '°C', 
            'speed': 'm/s'
        }
    },
    UnitSystem.B: {
        'area': 'acres',
        'distance': 'mile',
        'temperature': 'F', 
        'speed': 'ms',
        'labels': {
            'area': 'acres',
            'distance': 'mile',
            'temperature': '℉',
            'speed': 'm/s'
        }
    },
    UnitSystem.C: {
        'area': 'm2',
        'distance': 'km',
        'temperature': 'K',
        'speed': 'kmh', 
        'labels': {
            'area': 'm²',
            'distance': 'km',
            'temperature': 'K',
            'speed': 'km/h'
        }
    }
}


class UnitConverter:
    """Centralized unit conversion and formatting"""
    
    def __init__(self, unit_system: UnitSystem = UnitSystem.B):
        """Initialize with default unit system"""
        self.current_system = unit_system
        self.system_config = UNIT_SYSTEMS[unit_system]
    
    def set_unit_system(self, unit_system: Union[UnitSystem, str]) -> None:
        """Set the current unit system"""
        if isinstance(unit_system, str):
            unit_system = UnitSystem(unit_system)
        self.current_system = unit_system
        self.system_config = UNIT_SYSTEMS[unit_system]
    
    def get_current_system(self) -> UnitSystem:
        """Get current unit system"""
        return self.current_system
    
    def convert_area(self, value: float, from_unit: str = 'm2', to_unit: Optional[str] = None) -> float:
        """Convert area values"""
        if to_unit is None:
            to_unit = self.system_config['area']
        
        # Convert to base unit (m²) first
        base_value = value / CONVERSION_FACTORS['area'][from_unit]
        # Convert to target unit
        return base_value * CONVERSION_FACTORS['area'][to_unit]
    
    def convert_distance(self, value: float, from_unit: str = 'm', to_unit: Optional[str] = None) -> float:
        """Convert distance values"""
        if to_unit is None:
            to_unit = self.system_config['distance']
        
        # Convert to base unit (meters) first
        base_value = value / CONVERSION_FACTORS['distance'][from_unit]
        # Convert to target unit
        return base_value * CONVERSION_FACTORS['distance'][to_unit]
    
    def convert_temperature(self, value: float, from_unit: str = 'K', to_unit: Optional[str] = None) -> float:
        """Convert temperature values"""
        if to_unit is None:
            to_unit = self.system_config['temperature']
        
        # Convert from source unit to Kelvin first
        if from_unit == 'C':
            kelvin_value = value + 273.15
        elif from_unit == 'F':
            kelvin_value = (value - 32) * 5/9 + 273.15
        else:  # from_unit == 'K'
            kelvin_value = value
        
        # Convert from Kelvin to target unit
        if to_unit == 'K':
            return kelvin_value
        elif to_unit == 'C':
            return kelvin_value - 273.15
        elif to_unit == 'F':
            return (kelvin_value - 273.15) * 9/5 + 32
        
        return kelvin_value
    
    def convert_speed(self, value: float, from_unit: str = 'ms', to_unit: Optional[str] = None) -> float:
        """Convert speed values"""
        if to_unit is None:
            to_unit = self.system_config['speed']
        
        # Convert to base unit (m/s) first
        base_value = value / CONVERSION_FACTORS['speed'][from_unit]
        # Convert to target unit
        return base_value * CONVERSION_FACTORS['speed'][to_unit]
    
    def format_area(self, value_m2: float, precision: int = 2) -> str:
        """Format area value according to current unit system"""
        converted_value = self.convert_area(value_m2, 'm2')
        unit_label = self.system_config['labels']['area']
        return f"{converted_value:.{precision}f} {unit_label}"
    
    def format_distance(self, value_m: float, precision: int = 1) -> str:
        """Format distance value according to current unit system"""
        converted_value = self.convert_distance(value_m, 'm')
        unit_label = self.system_config['labels']['distance']
        return f"{converted_value:.{precision}f} {unit_label}"
    
    def format_temperature(self, value_k: float, precision: int = 1) -> str:
        """Format temperature value according to current unit system"""
        converted_value = self.convert_temperature(value_k, 'K')
        unit_label = self.system_config['labels']['temperature']
        return f"{converted_value:.{precision}f} {unit_label}"
    
    def format_speed(self, value_ms: float, precision: int = 1) -> str:
        """Format speed value according to current unit system"""
        converted_value = self.convert_speed(value_ms, 'ms')
        unit_label = self.system_config['labels']['speed']
        return f"{converted_value:.{precision}f} {unit_label}"
    
    def get_unit_label(self, unit_type: str) -> str:
        """Get unit label for current system"""
        return self.system_config['labels'].get(unit_type, '')
    
    def get_conversion_info(self) -> Dict[str, Any]:
        """Get current system conversion information"""
        return {
            'system': self.current_system.value,
            'units': self.system_config.copy(),
            'supported_conversions': list(CONVERSION_FACTORS.keys())
        }


def _initialize_from_config() -> UnitConverter:
    """Initialize unit converter from config.py settings"""
    try:
        from config import UNIT_SYSTEM_CONFIG
        current_system = UNIT_SYSTEM_CONFIG.get("current_system", "B")
        return UnitConverter(UnitSystem(current_system))
    except (ImportError, KeyError, ValueError) as e:
        # Fallback to default if config is not available or invalid
        print(f"Warning: Failed to load unit system from config ({e}), using default 'B'")
        return UnitConverter(UnitSystem.B)


# Global converter instance - initialized from config
_global_converter = _initialize_from_config()


def set_unit_system(unit_system: Union[UnitSystem, str]) -> None:
    """Set global unit system"""
    _global_converter.set_unit_system(unit_system)


def sync_with_config() -> None:
    """Synchronize global converter with current config.py settings"""
    try:
        from config import UNIT_SYSTEM_CONFIG
        current_system = UNIT_SYSTEM_CONFIG.get("current_system", "B")
        _global_converter.set_unit_system(UnitSystem(current_system))
    except (ImportError, KeyError, ValueError) as e:
        print(f"Warning: Failed to sync unit system with config ({e}), keeping current setting")


def get_current_system() -> UnitSystem:
    """Get current global unit system"""
    return _global_converter.get_current_system()


def get_converter() -> UnitConverter:
    """Get global converter instance"""
    return _global_converter


# Convenience functions using global converter
def convert_area(value: float, from_unit: str = 'm2', to_unit: Optional[str] = None) -> float:
    """Convert area using global converter"""
    return _global_converter.convert_area(value, from_unit, to_unit)


def convert_distance(value: float, from_unit: str = 'm', to_unit: Optional[str] = None) -> float:
    """Convert distance using global converter"""
    return _global_converter.convert_distance(value, from_unit, to_unit)


def convert_temperature(value: float, from_unit: str = 'K', to_unit: Optional[str] = None) -> float:
    """Convert temperature using global converter"""
    return _global_converter.convert_temperature(value, from_unit, to_unit)


def convert_speed(value: float, from_unit: str = 'ms', to_unit: Optional[str] = None) -> float:
    """Convert speed using global converter"""
    return _global_converter.convert_speed(value, from_unit, to_unit)


def format_area(value_m2: float, precision: int = 2) -> str:
    """Format area using global converter"""
    return _global_converter.format_area(value_m2, precision)


def format_distance(value_m: float, precision: int = 1) -> str:
    """Format distance using global converter"""
    return _global_converter.format_distance(value_m, precision)


def format_temperature(value_k: float, precision: int = 1) -> str:
    """Format temperature using global converter"""
    return _global_converter.format_temperature(value_k, precision)


def format_speed(value_ms: float, precision: int = 1) -> str:
    """Format speed using global converter"""
    return _global_converter.format_speed(value_ms, precision)


def get_unit_label(unit_type: str) -> str:
    """Get unit label using global converter"""
    return _global_converter.get_unit_label(unit_type)


def format_delta_with_units(prev_value: float, curr_value: float, 
                          unit_type: str, from_unit: str, 
                          precision: int = 1, pct_precision: int = 1, 
                          show_percent: bool = True) -> str:
    """
    Format delta with proper unit conversion and percentage.
    
    Args:
        prev_value: Previous value in original units
        curr_value: Current value in original units  
        unit_type: Type of unit ('area', 'distance', 'speed')
        from_unit: Original unit ('m2', 'm', 'ms', etc.)
        precision: Decimal precision for delta value
        pct_precision: Decimal precision for percentage
        show_percent: Whether to show percentage change
    
    Returns:
        Formatted delta string with units
    """
    # Convert both values to display units based on current system
    if unit_type == 'area':
        prev_converted = _global_converter.convert_area(prev_value, from_unit)
        curr_converted = _global_converter.convert_area(curr_value, from_unit)
    elif unit_type == 'distance':
        prev_converted = _global_converter.convert_distance(prev_value, from_unit)
        curr_converted = _global_converter.convert_distance(curr_value, from_unit)
    elif unit_type == 'speed':
        prev_converted = _global_converter.convert_speed(prev_value, from_unit)
        curr_converted = _global_converter.convert_speed(curr_value, from_unit)
    else:
        raise ValueError(f"Unsupported unit type: {unit_type}")
    
    # Calculate delta in converted units
    delta = curr_converted - prev_converted
    
    if not show_percent:
        delta_str = f"+{delta:.{precision}f}" if delta >= 0 else f"{delta:.{precision}f}"
        return delta_str
    
    # Calculate percentage change
    if prev_converted == 0:
        if curr_converted == 0:
            return f"0, 0.0%"
        else:
            return f"+{curr_converted:.{precision}f}, +∞%"
    
    pct_change = (delta / prev_converted) * 100
    delta_str = f"+{delta:.{precision}f}" if delta >= 0 else f"{delta:.{precision}f}"
    pct_str = f"+{pct_change:.{pct_precision}f}%" if pct_change >= 0 else f"{pct_change:.{pct_precision}f}%"
    
    return f"{delta_str}, {pct_str}"


def format_temperature_delta(prev_temp_k: float, curr_temp_k: float, 
                           precision: int = 1) -> str:
    """
    Format temperature delta (absolute change only, no percentage).
    Temperature percentage changes are not unit-invariant due to offset.
    
    Args:
        prev_temp_k: Previous temperature in Kelvin
        curr_temp_k: Current temperature in Kelvin
        precision: Decimal precision for delta value
        
    Returns:
        Formatted absolute temperature delta with units
    """
    # Convert both values to display units
    prev_converted = _global_converter.convert_temperature(prev_temp_k, 'K')
    curr_converted = _global_converter.convert_temperature(curr_temp_k, 'K')
    
    # Calculate absolute delta in converted units
    delta = curr_converted - prev_converted
    delta_str = f"+{delta:.{precision}f}" if delta >= 0 else f"{delta:.{precision}f}"
    
    return delta_str


# Note: Legacy compatibility functions removed
# All unit conversions should now use the centralized conversion functions
# with proper unit system configuration
