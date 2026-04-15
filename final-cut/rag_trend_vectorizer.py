#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Trend Vectorizer - Rolling Trend Feature Vectorizer

Specialized for no-fire RAG retrieval, using only rolling trend data for vectorization.
No environmental features (weather, terrain) when no fire points, only fire_overview.rolling trend data.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from config import RAG_CONFIG
from utils.llm_utils import safe_extract_number, safe_divide
from utils.date_utils import extract_date_cyclic_features


class TrendVectorizer:
    """Trend Feature Vectorizer"""
    
    def __init__(self):
        self.feature_names = None
        
    def vectorize_summary(self, summary: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert summary to trend feature vector

        Uses rolling trend data for similarity retrieval in no-fire scenarios.
        Includes fire rolling trends + resource rolling trends + cumulative context, no environmental features.

        Args:
            summary: Fire analysis summary data

        Returns:
            Trend feature vector, None if failed
        """
        try:
            features = []
            
            # === Rolling fire intensity features (8 dimensions) ===  
            rolling_fire_features = self._extract_rolling_fire_features(summary)
            features.extend(rolling_fire_features)
            
            # === Rolling resource trend features (8 dimensions) ===
            rolling_resource_features = self._extract_rolling_resource_features(summary)
            features.extend(rolling_resource_features)
            
            # === Cumulative context features (4 dimensions) ===
            cumulative_features = self._extract_cumulative_features(summary)
            features.extend(cumulative_features)
            
            # === Date cyclic features (2 dimensions) ===
            date_features = self._extract_date_features(summary)
            features.extend(date_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Failed to vectorize trend summary: {e}")
            return None
    
    
    def _extract_rolling_fire_features(self, summary: Dict[str, Any]) -> List[float]:
        """
        Extract rolling fire intensity features (only depends on fire_overview.rolling data).
        
        Args:
            summary: Fire analysis summary data
            
        Returns:
            List of 8 float features
        """
        features = []
        
        # Get rolling metrics from fire_overview.rolling
        fire_overview = summary.get("fire_overview", {})
        rolling_fire = fire_overview.get("rolling", {})
        
        if rolling_fire:
            # Prefer fire_points data
            fire_points_data = rolling_fire.get("fire_points", {})
            if fire_points_data:
                # FirePoints rolling features (4 dimensions)
                features.extend([
                    self._safe_float(fire_points_data.get("3day_avg_total")),
                    self._safe_float(fire_points_data.get("7day_avg_total")),
                    self._safe_float(fire_points_data.get("3day_max_total")),
                    self._safe_float(fire_points_data.get("current_vs_max_ratio"))
                ])
            else:
                # Fallback to old frp structure
                frp_data = rolling_fire.get("frp", {})
                features.extend([
                    self._safe_float(frp_data.get("3day_avg_total")),
                    self._safe_float(frp_data.get("7day_avg_total")),
                    self._safe_float(frp_data.get("3day_max_total")),
                    self._safe_float(frp_data.get("current_vs_max_ratio"))
                ])
            
            # Area rolling features (4 dimensions)
            area_data = rolling_fire.get("area", {})
            features.extend([
                self._safe_float(area_data.get("3day_avg_total")),
                self._safe_float(area_data.get("7day_avg_total")),
                self._safe_float(area_data.get("3day_max_total")),
                self._safe_float(area_data.get("current_vs_max_ratio"))
            ])
        else:
            # If no rolling data, fill with zeros (8 dimensions)
            features.extend([0.0] * 8)
        
        return features
    
    def _extract_rolling_resource_features(self, summary: Dict[str, Any]) -> List[float]:
        """
        Extract rolling resource trend features (based on cumulative_context_info.rolling_stats).
        
        Args:
            summary: Fire analysis summary data
            
        Returns:
            List of 8 float features
        """
        features = []
        
        # Get rolling statistics from cumulative_context_info
        cumulative_info = summary.get("cumulative_context_info", {})
        rolling_stats = cumulative_info.get("rolling_stats", {})
        
        # 3-day rolling average
        day3_stats = rolling_stats.get("3day", {})
        features.extend([
            self._safe_float(day3_stats.get("avg_daily_cost", 0)),
            self._safe_float(day3_stats.get("avg_daily_personnel", 0))
        ])
        
        # 7-day rolling average  
        day7_stats = rolling_stats.get("7day", {})
        features.extend([
            self._safe_float(day7_stats.get("avg_daily_cost", 0)),
            self._safe_float(day7_stats.get("avg_daily_personnel", 0))
        ])
        
        # Cumulative metrics
        features.extend([
            self._safe_float(cumulative_info.get("total_cumulative_cost", 0)),
            self._safe_float(cumulative_info.get("total_cumulative_personnel_days", 0)),
            self._safe_float(cumulative_info.get("days_since_fire_start", 0)),
            # Resource intensity ratio (cost/days)
            safe_divide(
                cumulative_info.get("total_cumulative_cost", 0),
                cumulative_info.get("days_since_fire_start", 1)
            )
        ])
        
        return features
    
    def _extract_cumulative_features(self, summary: Dict[str, Any]) -> List[float]:
        """
        Extract cumulative context features (simplified version).
        
        Args:
            summary: Fire analysis summary data
            
        Returns:
            List of 4 float features
        """
        cumulative_info = summary.get("cumulative_context_info", {})
        
        return [
            self._safe_float(cumulative_info.get("total_cumulative_cost", 0)),
            self._safe_float(cumulative_info.get("total_cumulative_personnel_days", 0)),
            self._safe_float(cumulative_info.get("days_since_fire_start", 0)),
            # Average daily resource intensity
            safe_divide(
                cumulative_info.get("total_cumulative_personnel_days", 0),
                cumulative_info.get("days_since_fire_start", 1)
            )
        ]
    
    def _extract_date_features(self, summary: Dict[str, Any]) -> List[float]:
        """
        Extract date cyclic features (sine/cosine encoding).
        
        Args:
            summary: Fire analysis summary data
            
        Returns:
            List of two floats: [date_sin, date_cos]
        """
        analysis_mmdd = summary.get("analysis_mmdd", "")
        return extract_date_cyclic_features(analysis_mmdd)
    
    def _safe_float(self, value: Any) -> float:
        """
        Safe float conversion (wrapper for safe_extract_number).
        
        Args:
            value: Value to convert
            
        Returns:
            Float representation, 0.0 if conversion fails
        """
        return safe_extract_number(value, default=0.0, as_int=False)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature name list.
        
        Returns:
            List of feature names (22 total)
        """
        if self.feature_names is not None:
            return self.feature_names
        
        names = []
        
        # Rolling fire intensity features (8 dimensions) - FirePoints + Area
        names.extend([
            "rolling3_avg_fire_points", "rolling7_avg_fire_points",
            "rolling3_max_fire_points", "current_vs_max_fire_points_ratio",
            "rolling3_avg_area", "rolling7_avg_area", 
            "rolling3_max_area", "current_vs_max_area_ratio"
        ])
        
        # Rolling resource trend features (8 dimensions) - Cost + Personnel + Cumulative
        names.extend([
            "rolling3_avg_daily_cost", "rolling3_avg_daily_personnel",
            "rolling7_avg_daily_cost", "rolling7_avg_daily_personnel", 
            "total_cumulative_cost", "total_cumulative_personnel_days",
            "days_since_fire_start", "avg_daily_cost_intensity"
        ])
        
        # Cumulative context features (4 dimensions)
        names.extend([
            "cumulative_cost", "cumulative_personnel_days",
            "fire_duration_days", "avg_daily_personnel_intensity"
        ])
        
        # Date cyclic features (2 dimensions)
        names.extend(["date_sin", "date_cos"])
        
        return names
