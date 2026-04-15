#!/usr/bin/env python3
"""
Terrain Analysis Module for NLCD Land Cover Data
Provides comprehensive terrain analysis including patch metrics, DEM analysis, and natural language descriptions
"""

import numpy as np
# Removed scipy.stats import as we now calculate diversity directly
from scipy import ndimage
from typing import Dict, Any, List, Tuple
from db_fetcher import DBDataFetcher
from database import get_db_cursor
from config import TABLE_NAMES, SRID_CONFIG
from utils.result_builders import build_error_result


# NLCD Land Cover Classifications
NLCD_LABELS = {
    11: "Open Water", 12: "Perennial Ice/Snow",
    21: "Developed, Open Space", 22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity", 24: "Developed, High Intensity",
    31: "Barren Land", 41: "Deciduous Forest", 42: "Evergreen Forest",
    43: "Mixed Forest", 52: "Shrub/Scrub", 71: "Grassland/Herbaceous",
    81: "Pasture/Hay", 82: "Cultivated Crops",
    90: "Woody Wetlands", 95: "Emergent Herbaceous Wetlands"
}

NLCD_DESCRIPTIONS = {
    11: "Areas of open water, generally with less than 25% cover of vegetation or soil.",
    12: "Areas characterized by perennial cover of ice/snow, generally greater than 25% of total cover.",
    21: "Areas with constructed materials and vegetation. Large-lot housing, parks, golf courses. Impervious surfaces <20%.",
    22: "Areas with constructed materials and vegetation. Single-family housing. Impervious surfaces 20-49%.",
    23: "Areas with constructed materials and vegetation. Single-family housing. Impervious surfaces 50-79%.",
    24: "Highly developed areas. Apartments, commercial/industrial. Impervious surfaces 80-100%.",
    31: "Bedrock, desert pavement, sand dunes, strip mines, gravel pits. Vegetation <15%.",
    41: "Trees >5m tall, >20% vegetation cover. >75% deciduous species shed foliage seasonally.",
    42: "Trees >5m tall, >20% vegetation cover. >75% evergreen species maintain leaves year-round.",
    43: "Trees >5m tall, >20% vegetation cover. Neither deciduous nor evergreen >75%.",
    52: "Shrubs <5m tall, shrub canopy >20% vegetation. Includes young trees and stunted trees.",
    71: "Graminoid/herbaceous vegetation >80%. Not intensively managed, can be grazed.",
    81: "Grasses, legumes planted for livestock grazing or hay production. >20% vegetation.",
    82: "Annual crops (corn, soybeans, vegetables) and perennial woody crops. >20% vegetation.",
    90: "Forest/shrubland >20% cover where soil is periodically saturated with water.",
    95: "Perennial herbaceous vegetation >80% where soil is periodically saturated with water."
}


class TerrainAnalyzer:
    """Comprehensive terrain analysis for fire risk assessment"""
    
    def __init__(self):
        self.fetcher = DBDataFetcher()
    
    def get_nlcd_data_for_polygon(self, polygon_wkt: str) -> Dict[str, Any]:
        """
        Get NLCD raster data for a given polygon.
        
        Args:
            polygon_wkt: Well-Known Text representation of polygon
            
        Returns:
            Dictionary containing NLCD data or error information
        """
        try:
            return self.fetcher.get_nlcd_stats(polygon_wkt)
        except Exception as e:
            return {"error": f"Failed to get NLCD data: {str(e)}"}
    
    def extract_land_cover_stats(self, nlcd_values: Any) -> List[Dict[str, Any]]:
        """
        Extract land cover statistics from NLCD raster values.
        
        Args:
            nlcd_values: NLCD raster values from database (various formats supported)
            
        Returns:
            List of dictionaries containing land cover statistics, sorted by percentage
        """
        if not hasattr(nlcd_values, '__iter__'):
            return []
        
        try:
            values_array = self._extract_values_array(nlcd_values)
            valid_values = self._clean_raster_values(values_array)
            
            if len(valid_values) == 0:
                return []
            
            type_stats = self._build_land_cover_stats(valid_values)
            type_stats.sort(key=lambda x: x['percent'], reverse=True)
            return type_stats
            
        except Exception as e:
            print(f"Error extracting land cover stats: {e}")
            return []
    
    def _extract_values_array(self, nlcd_values: Any) -> np.ndarray:
        """
        Extract values array from various NLCD data formats.
        
        Args:
            nlcd_values: NLCD data in various possible formats
            
        Returns:
            Flattened numpy array of values
        """
        if hasattr(nlcd_values, 'valarray'):
            values_array = nlcd_values.valarray
        else:
            values_array = nlcd_values
        
        return np.array(values_array, dtype=float).flatten()
    
    def _clean_raster_values(self, values_flat: np.ndarray) -> np.ndarray:
        """
        Remove no-data values from raster array.
        
        Args:
            values_flat: Flattened array of raster values
            
        Returns:
            Array with valid values only (no zeros, NaNs, or None)
        """
        try:
            valid_mask = np.logical_and(values_flat != 0, ~np.isnan(values_flat))
            return values_flat[valid_mask]
        except (TypeError, ValueError):
            # Fallback for non-numeric data
            return np.array([
                v for v in values_flat 
                if v is not None and v != 0 and not (isinstance(v, float) and np.isnan(v))
            ])
    
    def _build_land_cover_stats(self, valid_values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Build land cover statistics from valid raster values.
        
        Args:
            valid_values: Array of valid NLCD values
            
        Returns:
            List of dictionaries with land cover statistics
        """
        unique_vals, counts = np.unique(valid_values, return_counts=True)
        total_count = counts.sum()
        
        type_stats = []
        for val, count in zip(unique_vals, counts):
            land_code = int(val)
            if land_code in NLCD_LABELS:
                type_stats.append({
                    'code': land_code,
                    'name': NLCD_LABELS[land_code],
                    'count': int(count),
                    'percent': round((count / total_count) * 100, 1)
                })
        
        return type_stats
    
    def calculate_landscape_metrics(self, land_cover_stats: List[Dict[str, Any]], total_pixels: int) -> Dict[str, Any]:
        """
        Calculate landscape diversity and fragmentation metrics.
        
        Uses Shannon diversity index and dominant type percentage to assess
        landscape heterogeneity and fragmentation.
        
        Args:
            land_cover_stats: List of land cover statistics from extract_land_cover_stats
            total_pixels: Total number of valid pixels
            
        Returns:
            Dictionary containing landscape metrics (num_land_types, dominant_type_percent,
            diversity_index, fragmentation_level)
        """
        if not land_cover_stats or total_pixels == 0:
            return {
                'num_land_types': 0,
                'dominant_type_percent': 0,
                'diversity_index': 0,
                'fragmentation_level': 'unknown'
            }
        
        num_types = len(land_cover_stats)
        dominant_percent = land_cover_stats[0]['percent'] if land_cover_stats else 0
        diversity_index = self._calculate_shannon_diversity(land_cover_stats)
        fragmentation_level = self._assess_fragmentation_level(dominant_percent)
        
        return {
            'num_land_types': num_types,
            'dominant_type_percent': dominant_percent,
            'diversity_index': round(diversity_index, 3),
            'fragmentation_level': fragmentation_level
        }
    
    def _calculate_shannon_diversity(self, land_cover_stats: List[Dict[str, Any]]) -> float:
        """
        Calculate Shannon diversity index.
        
        Args:
            land_cover_stats: List of land cover statistics
            
        Returns:
            Shannon diversity index value
        """
        diversity_index = 0
        for stat in land_cover_stats:
            p = stat['percent'] / 100
            if p > 0:
                diversity_index -= p * np.log(p)
        return diversity_index
    
    def _assess_fragmentation_level(self, dominant_percent: float) -> str:
        """
        Assess landscape fragmentation level based on dominant type percentage.
        
        Args:
            dominant_percent: Percentage of dominant land cover type
            
        Returns:
            Fragmentation level: 'low', 'moderate', or 'high'
        """
        if dominant_percent > 80:
            return 'low'
        elif dominant_percent > 50:
            return 'moderate'
        else:
            return 'high'
    
    def analyze_vegetation_fire_risk(self, land_cover_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze fire risk based on vegetation and land cover types.
        
        Classifies land cover types into high/moderate/low fire risk categories
        based on NLCD codes and calculates overall risk level.
        
        Args:
            land_cover_stats: List of land cover statistics
            
        Returns:
            Dictionary containing fire risk metrics (high/moderate/low risk percentages,
            risk types, and overall risk level)
        """
        # Fire risk classification based on NLCD codes
        high_risk_codes = [41, 42, 43, 52, 71]  # Forests, shrubs, grasslands
        moderate_risk_codes = [81, 82, 21, 22]  # Agriculture, low-density development
        low_risk_codes = [11, 90, 95, 23, 24, 31]  # Water, wetlands, high-density development, barren
        
        high_risk_percent = 0
        moderate_risk_percent = 0
        low_risk_percent = 0
        high_risk_types = []
        moderate_risk_types = []
        low_risk_types = []
        
        for stat in land_cover_stats:
            code = stat['code']
            percent = stat['percent']
            name = stat['name']
            
            if code in high_risk_codes:
                high_risk_percent += percent
                high_risk_types.append(f"{name} ({percent}%)")
            elif code in moderate_risk_codes:
                moderate_risk_percent += percent
                moderate_risk_types.append(f"{name} ({percent}%)")
            elif code in low_risk_codes:
                low_risk_percent += percent
                low_risk_types.append(f"{name} ({percent}%)")
        
        # Determine overall risk level
        if high_risk_percent > 60:
            overall_risk = "Very High"
        elif high_risk_percent > 40:
            overall_risk = "High"
        elif high_risk_percent > 20:
            overall_risk = "Moderate"
        else:
            overall_risk = "Low"
        
        return {
            'high_risk_percent': round(high_risk_percent, 1),
            'moderate_risk_percent': round(moderate_risk_percent, 1),
            'low_risk_percent': round(low_risk_percent, 1),
            'high_risk_types': high_risk_types,
            'moderate_risk_types': moderate_risk_types,
            'low_risk_types': low_risk_types,
            'overall_risk': overall_risk
        }
    
    def analyze_fire_spread_potential(self, land_cover_stats: List[Dict[str, Any]], landscape_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze fire spread potential based on fuel continuity and barriers.
        
        Evaluates continuous fuel coverage, natural and artificial barriers,
        and assesses overall spread potential and barrier effectiveness.
        
        Args:
            land_cover_stats: List of land cover statistics
            landscape_metrics: Landscape metrics from calculate_landscape_metrics
            
        Returns:
            Dictionary containing spread analysis (continuous_fuel_percent, barriers,
            spread_potential, barrier_effectiveness, fragmentation_level)
        """
        # Identify fuel types and barriers
        continuous_fuel_percent = 0
        natural_barriers_percent = 0
        artificial_barriers_percent = 0
        
        # Continuous fuel types (high fire spread potential)
        continuous_fuel_codes = [41, 42, 43, 52, 71]  # Forests, shrubs, grasslands
        # Natural barriers (low fire spread)
        natural_barrier_codes = [11, 90, 95, 31]  # Water, wetlands, barren land
        # Artificial barriers (moderate fire resistance)
        artificial_barrier_codes = [21, 22, 23, 24, 81, 82]  # Development, agriculture
        
        for stat in land_cover_stats:
            code = stat['code']
            percent = stat['percent']
            
            if code in continuous_fuel_codes:
                continuous_fuel_percent += percent
            elif code in natural_barrier_codes:
                natural_barriers_percent += percent
            elif code in artificial_barrier_codes:
                artificial_barriers_percent += percent
        
        # Assess spread potential
        if continuous_fuel_percent > 70:
            spread_potential = "Very High"
        elif continuous_fuel_percent > 50:
            spread_potential = "High"
        elif continuous_fuel_percent > 30:
            spread_potential = "Moderate"
        else:
            spread_potential = "Low"
        
        # Barrier effectiveness
        total_barriers = natural_barriers_percent + artificial_barriers_percent
        if total_barriers > 30:
            barrier_effectiveness = "High"
        elif total_barriers > 15:
            barrier_effectiveness = "Moderate"
        else:
            barrier_effectiveness = "Low"
        
        return {
            'continuous_fuel_percent': round(continuous_fuel_percent, 1),
            'natural_barriers_percent': round(natural_barriers_percent, 1),
            'artificial_barriers_percent': round(artificial_barriers_percent, 1),
            'spread_potential': spread_potential,
            'barrier_effectiveness': barrier_effectiveness,
            'fragmentation_level': landscape_metrics.get('fragmentation_level', 'unknown')
        }
    
    def generate_natural_description(self, nlcd_data: Dict[str, Any], fire_context: bool = True) -> str:
        """
        Generate comprehensive natural language description from NLCD data.
        
        Creates a human-readable narrative describing land cover composition,
        fire risk assessment, and landscape characteristics.
        
        Args:
            nlcd_data: NLCD raster data dictionary
            fire_context: Whether to include fire-specific analysis
            
        Returns:
            Natural language description string
        """
        if 'error' in nlcd_data:
            return f"Unable to analyze terrain: {nlcd_data['error']}"
        
        land_cover_stats = self.extract_land_cover_stats(nlcd_data.get('values'))
        
        if not land_cover_stats:
            return "Unable to analyze land cover composition in this area - no valid NLCD data found."
        
        total_pixels = sum(stat['count'] for stat in land_cover_stats)
        landscape_metrics = self.calculate_landscape_metrics(land_cover_stats, total_pixels)
        
        description_parts = []
        
        # Land cover composition
        description_parts.append(self._build_composition_description(land_cover_stats))
        
        # Fire-specific analysis
        if fire_context:
            fire_risk = self.analyze_vegetation_fire_risk(land_cover_stats)
            spread_analysis = self.analyze_fire_spread_potential(land_cover_stats, landscape_metrics)
            description_parts.extend(
                self._build_fire_analysis_descriptions(fire_risk, spread_analysis)
            )
        
        # Landscape diversity
        description_parts.append(self._build_diversity_description(landscape_metrics))
        
        return " ".join(description_parts)
    
    def _build_composition_description(self, land_cover_stats: List[Dict[str, Any]]) -> str:
        """
        Build land cover composition description.
        
        Args:
            land_cover_stats: List of land cover statistics
            
        Returns:
            Composition description string
        """
        top_3_types = land_cover_stats[:3]
        composition_str = ", ".join([f"{stat['name']} ({stat['percent']}%)" for stat in top_3_types])
        return (
            f"Based on NLCD satellite data analysis, this fire-affected area "
            f"is primarily composed of {composition_str}."
        )
    
    def _build_fire_analysis_descriptions(self, fire_risk: Dict[str, Any], 
                                         spread_analysis: Dict[str, Any]) -> List[str]:
        """
        Build fire risk and spread analysis descriptions.
        
        Args:
            fire_risk: Fire risk analysis results
            spread_analysis: Fire spread analysis results
            
        Returns:
            List of description strings
        """
        descriptions = []
        
        # High-risk vegetation
        if fire_risk['high_risk_percent'] > 0:
            high_risk_desc = f"High-risk flammable vegetation covers {fire_risk['high_risk_percent']}% of the area"
            if fire_risk['high_risk_types']:
                high_risk_desc += f", consisting of {', '.join(fire_risk['high_risk_types'][:2])}"
            descriptions.append(high_risk_desc + ".")
        
        # Fire spread potential
        spread_desc = f"Fire spread potential is {spread_analysis['spread_potential'].lower()}"
        if spread_analysis['continuous_fuel_percent'] > 50:
            spread_desc += f" due to {spread_analysis['continuous_fuel_percent']}% continuous fuel coverage"
        descriptions.append(spread_desc + ".")
        
        # Natural barriers
        if fire_risk['low_risk_percent'] > 10:
            barriers_desc = f"Natural fire barriers account for {fire_risk['low_risk_percent']}% of the area"
            if fire_risk['low_risk_types']:
                barriers_desc += f", including {', '.join(fire_risk['low_risk_types'][:2])}"
            barriers_desc += f", with {spread_analysis['barrier_effectiveness'].lower()} barrier effectiveness"
            descriptions.append(barriers_desc + ".")
        
        # Coverage summary
        if fire_risk['high_risk_percent'] > 0:
            coverage_summary = f"Total high-risk vegetation coverage: {fire_risk['high_risk_percent']}% of area"
            descriptions.append(coverage_summary + ".")
        
        return descriptions
    
    def _build_diversity_description(self, landscape_metrics: Dict[str, Any]) -> str:
        """
        Build landscape diversity description.
        
        Args:
            landscape_metrics: Landscape metrics from calculate_landscape_metrics
            
        Returns:
            Diversity description string
        """
        return (
            f"The landscape shows {landscape_metrics['fragmentation_level']} fragmentation "
            f"with {landscape_metrics['num_land_types']} distinct land cover types "
            f"and a diversity index of {landscape_metrics['diversity_index']}."
        )
    
    def analyze_cluster_terrain(self, cluster_polygon_wkt: str) -> Dict[str, Any]:
        """
        Analyze terrain for a fire cluster polygon.
        
        Performs comprehensive terrain analysis including NLCD data extraction,
        land cover statistics, fire risk assessment, and metric calculation.
        
        Args:
            cluster_polygon_wkt: Well-Known Text representation of cluster polygon
            
        Returns:
            Dictionary containing natural description, quantitative metrics, and fire context
        """
        nlcd_data = self.get_nlcd_data_for_polygon(cluster_polygon_wkt)
        
        if 'error' in nlcd_data:
            return self._build_error_result(nlcd_data)
        
        land_cover_stats = self.extract_land_cover_stats(nlcd_data.get('values'))
        
        if not land_cover_stats:
            return self._build_no_data_result(nlcd_data)
        
        quantitative_metrics = self._calculate_all_terrain_metrics(land_cover_stats)
        natural_description = self.generate_natural_description(nlcd_data, fire_context=True)
        
        return {
            'natural_description': natural_description,
            'fire_context': True,
            'quantitative_metrics': quantitative_metrics
        }
    
    def _build_error_result(self, nlcd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build error result dictionary."""
        base_data = {
            'nlcd_analysis': nlcd_data,
            'natural_description': f"Unable to analyze terrain: {nlcd_data['error']}",
            'fire_context': True
        }
        return build_error_result(nlcd_data['error'], base_data)
    
    def _build_no_data_result(self, nlcd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build no-data result dictionary."""
        base_data = {
            'nlcd_analysis': nlcd_data,
            'natural_description': "Unable to analyze land cover composition in this area - no valid NLCD data found.",
            'fire_context': True
        }
        return build_error_result('No valid NLCD data found', base_data)
    
    def _calculate_all_terrain_metrics(self, land_cover_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate all terrain metrics for ML features.
        
        Args:
            land_cover_stats: List of land cover statistics
            
        Returns:
            Dictionary containing all quantitative metrics
        """
        total_pixels = sum(stat['count'] for stat in land_cover_stats)
        landscape_metrics = self.calculate_landscape_metrics(land_cover_stats, total_pixels)
        fire_risk_metrics = self.analyze_vegetation_fire_risk(land_cover_stats)
        fire_spread_metrics = self.analyze_fire_spread_potential(land_cover_stats, landscape_metrics)
        
        return {
            # Landscape metrics for ML features
            'num_land_types': landscape_metrics['num_land_types'],
            'dominant_type_percent': landscape_metrics['dominant_type_percent'],
            'diversity_index': landscape_metrics['diversity_index'],
            'fragmentation_level': landscape_metrics['fragmentation_level'],
            
            # Fire risk metrics for ML features  
            'high_risk_percent': fire_risk_metrics['high_risk_percent'],
            'moderate_risk_percent': fire_risk_metrics['moderate_risk_percent'],
            'low_risk_percent': fire_risk_metrics['low_risk_percent'],
            'overall_risk_score': self._convert_risk_to_score(fire_risk_metrics['overall_risk']),
            
            # Fire spread metrics for ML features
            'continuous_fuel_percent': fire_spread_metrics['continuous_fuel_percent'],
            'natural_barriers_percent': fire_spread_metrics['natural_barriers_percent'],
            'spread_potential_score': self._convert_spread_to_score(fire_spread_metrics['spread_potential']),
            
            # Summary statistics
            'total_pixels': total_pixels
        }
    
    def _convert_level_to_score(self, level: str) -> int:
        """
        Convert risk/spread level to numerical score (unified method).
        
        Args:
            level: Level string (Very Low/Low/Moderate/High/Very High)
            
        Returns:
            Numerical score (1-5), defaults to 3 for unknown levels
        """
        level_scores = {
            'Very Low': 1,
            'Low': 2,
            'Moderate': 3,
            'High': 4,
            'Very High': 5
        }
        return level_scores.get(level, 3)  # Default to moderate
    
    # Backward compatibility aliases
    def _convert_risk_to_score(self, risk_level: str) -> int:
        """Legacy alias - use _convert_level_to_score instead"""
        return self._convert_level_to_score(risk_level)
    
    def _convert_spread_to_score(self, spread_potential: str) -> int:
        """Legacy alias - use _convert_level_to_score instead"""
        return self._convert_level_to_score(spread_potential)