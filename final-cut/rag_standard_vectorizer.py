#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Standard Vectorizer - Instant Feature-based Vectorizer

For RAG retrieval with active fire points, using instant features for vectorization.
Includes fire overview, affected areas, fire stations, weather, terrain, etc.
"""

import numpy as np
from typing import Dict, Any, Optional

from config import RAG_CONFIG
from utils.date_utils import extract_date_cyclic_features


class StandardVectorizer:
    """Standard Feature Vectorizer"""
    
    def __init__(self):
        self.feature_names = None
        
    def vectorize_summary(self, summary: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert summary to standard feature vector.

        Uses instant feature data for similarity retrieval with active fires.
        Includes fire overview + affected areas + fire stations + weather + terrain + date features.

        Field constraints:
        - Does not use affected_areas.counties
        - fire_stations only uses total_stations, nearest_distance_m
        - Includes date cyclic features
        
        Args:
            summary: Fire analysis summary data
            
        Returns:
            Standard feature vector, None if failed
        """
        try:
            features = []
            
            # === fire_overview features (6 dimensions) ===
            fire_overview = summary.get('fire_overview', {})
            features.extend([
                float(fire_overview.get('total_fire_points', 0)),
                float(fire_overview.get('num_clusters', 0)),
                float(fire_overview.get('total_frp', 0)),
                float(fire_overview.get('total_area_m2', 0)),
                float(fire_overview.get('max_brightness', 0)),
                float(fire_overview.get('max_frp', 0))
            ])
            
            # === affected_areas features (2 dimensions, excluding counties) ===
            affected_areas = summary.get('affected_areas', {})
            features.extend([
                float(affected_areas.get('total_population_affected', 0)),
                float(affected_areas.get('num_counties', 0))
            ])
            
            # === fire_stations features (2 dimensions) ===
            fire_stations = summary.get('fire_stations', {})
            features.extend([
                float(fire_stations.get('total_stations', 0)),
                float(fire_stations.get('nearest_distance_m', 0) or 0)
            ])
            
            # === weather_conditions features (5 dimensions) ===
            weather = summary.get('weather_conditions', {})
            features.extend([
                float(weather.get('bi', 0) or 0),
                float(weather.get('tmmx', 0) or 0),
                float(weather.get('tmmn', 0) or 0),
                float(weather.get('vs', 0) or 0),
                float(weather.get('fm1', 0) or 0)
            ])
            
            # === terrain_conditions features (10 dimensions) ===
            terrain = summary.get('terrain_conditions', {})
            features.extend([
                float(terrain.get('avg_num_land_types', 0) or 0),
                float(terrain.get('avg_dominant_type_percent', 0) or 0),
                float(terrain.get('avg_diversity_index', 0) or 0),
                float(terrain.get('avg_high_risk_percent', 0) or 0),
                float(terrain.get('avg_moderate_risk_percent', 0) or 0),
                float(terrain.get('avg_low_risk_percent', 0) or 0),
                float(terrain.get('avg_overall_risk_score', 0) or 0),
                float(terrain.get('avg_continuous_fuel_percent', 0) or 0),
                float(terrain.get('avg_natural_barriers_percent', 0) or 0),
                float(terrain.get('avg_spread_potential_score', 0) or 0)
            ])
            
            # === Date cyclic features (2 dimensions) ===
            date_features = self._extract_date_features(summary)
            features.extend(date_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Failed to vectorize summary: {e}")
            return None
    
    def _extract_date_features(self, summary: Dict[str, Any]) -> list:
        """
        Extract date cyclic features (sine/cosine encoding).
        
        Args:
            summary: Fire analysis summary data
            
        Returns:
            List of two floats: [date_sin, date_cos]
        """
        analysis_mmdd = summary.get('analysis_mmdd', '')
        return extract_date_cyclic_features(analysis_mmdd)
    
    def get_feature_names(self) -> list:
        """
        Get feature name list.
        
        Returns:
            List of feature names (27 total)
        """
        return [
            # fire_overview (6)
            'total_fire_points', 'num_clusters', 'total_frp', 'total_area_m2', 
            'max_brightness', 'max_frp',
            # affected_areas (2)
            'total_population_affected', 'num_counties',
            # fire_stations (2)
            'total_stations', 'nearest_distance_m',
            # weather_conditions (5)
            'bi', 'tmmx', 'tmmn', 'vs', 'fm1',
            # terrain_conditions (10)
            'avg_num_land_types', 'avg_dominant_type_percent', 'avg_diversity_index',
            'avg_high_risk_percent', 'avg_moderate_risk_percent', 'avg_low_risk_percent',
            'avg_overall_risk_score', 'avg_continuous_fuel_percent', 
            'avg_natural_barriers_percent', 'avg_spread_potential_score',
            # date features (2)
            'date_sin', 'date_cos'
        ]
