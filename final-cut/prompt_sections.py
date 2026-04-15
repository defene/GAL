#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Sections - Modular User Prompt Components

Defines Context data structure and section builder functions for user prompts.
Each section is independent and can be enabled/disabled or reordered via configuration.
"""

from typing import Any, Dict, List, Optional, Callable, Protocol
from dataclasses import dataclass
import json
import numpy as np

from utils.unit_converter import get_converter, format_area, format_distance, format_temperature, format_speed, format_delta_with_units
from utils.llm_utils import safe_extract_number
from utils.comparison_utils import format_value_with_arrow, compare_lists
from config import UNIT_SYSTEM_CONFIG


def format_nan_value(value: float, format_str: str = "{:.1f}", na_text: str = "N/A") -> str:
    """
    Format a numeric value that may contain NaN.
    
    Args:
        value: Numeric value to format (may be NaN)
        format_str: Format string to apply to valid values
        na_text: Text to display for NaN values
        
    Returns:
        Formatted string representation
    """
    if np.isnan(value):
        return na_text
    return format_str.format(value)


def _should_include_environmental_info(ctx) -> bool:
    """Check if environmental information should be included based on configuration"""
    from config import PROMPT_SECTION_CONFIG
    
    # Get configuration from current template
    current_template = PROMPT_SECTION_CONFIG.get("current_template", "template1")
    template_config = PROMPT_SECTION_CONFIG.get("templates", {}).get(current_template, {})
    fire_analysis_config = template_config.get("fire_analysis", {})
    user_config = fire_analysis_config.get("user", {})
    toggles = user_config.get("toggles", {})
    
    return toggles.get("include_environmental_info", True)


def _build_cluster_line(cluster: Dict[str, Any], ctx) -> str:
    """Build cluster line with optional environmental information"""
    try:
        cluster_id = str(cluster.get("cluster_id", "unknown"))
        
        # Extract cluster info
        cluster_info = cluster.get("cluster_info", {})
        points = int(cluster_info.get("points", 0))
        frp = safe_extract_number(cluster_info.get("frp"), default=np.nan, as_int=False)
        max_brightness = safe_extract_number(cluster_info.get("max_brightness"), default=np.nan, as_int=False)
        area_m2 = safe_extract_number(cluster_info.get("area_m2"), default=np.nan, as_int=False)
        
        # Extract location context
        county_data = cluster.get("county", {})
        county_name = county_data.get("county_name", "Unknown")
        state = county_data.get("state", "Unknown")
        
        # Format displays with unit conversion
        area_display = format_area(area_m2, precision=2) if not np.isnan(area_m2) else "N/A"
        brightness_display = format_nan_value(max_brightness, "{:.1f} K")
        
        # Build cluster line based on environmental info settings
        if _should_include_environmental_info(ctx):
            # Extract population
            population_data = cluster.get("population", {})
            population = int(population_data.get("pop_sum", 0))
            
            # Extract fire stations
            stations_data = cluster.get("fire_stations", {})
            stations = stations_data.get("stations", [])
            
            # Format fire station distances (limit to first 5)
            station_distances = []
            for i, dist in enumerate(stations[:5]):
                dist_float = safe_extract_number(dist, default=np.nan, as_int=False)
                if not np.isnan(dist_float):
                    station_distances.append(f"station_{i+1}={format_distance(dist_float, precision=1)}")
            stations_str = ", ".join(station_distances) if station_distances else "no_stations"
            
            # Extract weather conditions
            weather_data = cluster.get("weather", {})
            weather_values = weather_data.get("values", {})
            
            bi = safe_extract_number(weather_values.get("bi"), default=np.nan, as_int=False)
            tmmx = safe_extract_number(weather_values.get("tmmx"), default=np.nan, as_int=False)
            tmmn = safe_extract_number(weather_values.get("tmmn"), default=np.nan, as_int=False)
            vs = safe_extract_number(weather_values.get("vs"), default=np.nan, as_int=False)
            fm1 = safe_extract_number(weather_values.get("fm1"), default=np.nan, as_int=False)
            
            # Extract terrain description
            terrain_description = _format_terrain_description(cluster, ctx)
            
            tmmx_display = format_temperature(tmmx, precision=1) if not np.isnan(tmmx) else "N/A"
            tmmn_display = format_temperature(tmmn, precision=1) if not np.isnan(tmmn) else "N/A"
            speed_display = format_speed(vs, precision=1) if not np.isnan(vs) else "N/A"
            
            return (
                f"- Cluster {cluster_id}: "
                f"fire[points={points}, frp={format_nan_value(frp, '{:.1f} MW')}, brightness={brightness_display}, area={area_display}], "
                f"weather[BI={format_nan_value(bi)}, tmax={tmmx_display}, tmin={tmmn_display}, wind={speed_display}, FM1={format_nan_value(fm1, '{:.1f}%')}], "
                f"location[{county_name}, {state}, pop={population:.0f}, {stations_str}], "
                f"terrain: {terrain_description}"
            )
        else:
            return (
                f"- Cluster {cluster_id}: "
                f"fire[points={points}, frp={format_nan_value(frp, '{:.1f} MW')}, brightness={brightness_display}, area={area_display}]"
            )
    except Exception as e:
        print(f"Warning: Failed to format cluster {cluster.get('cluster_id', 'unknown')}: {e}")
        return f"- Cluster {cluster.get('cluster_id', 'unknown')}: Error formatting cluster data"


@dataclass
class Context:
    """Context data for prompt building - consumes pre-calculated analysis results"""
    # Core inputs
    summary: Dict[str, Any]
    clusters_data: List[Dict[str, Any]]
    task_mode: str = "fire_analysis"
    previous_analysis: Optional[Dict[str, Any]] = None
    previous_summary: Optional[Dict[str, Any]] = None
    rag_context: Optional[str] = None
    
    # Unit system for display formatting
    unit_system: str = "B"  # Default to system B, will be overridden from config
    
    def __post_init__(self):
        """Initialize unit system from config"""
        self.unit_system = UNIT_SYSTEM_CONFIG.get("current_system", "B")


class SectionBuilder(Protocol):
    """Protocol for section builder functions"""
    
    def __call__(self, ctx: Context) -> str:
        """Build section content from context"""
        ...


# =============================================================================
# User Prompt Section Builders
# =============================================================================

def fire_overview(ctx: Context) -> str:
    """Build fire overview section from pre-calculated statistics with optional comparison"""
    # Check if we have previous data for comparison
    has_previous = bool(ctx.previous_summary)
    
    if has_previous:
        lines = ["## Fire Overview vs Yesterday"]
    else:
        lines = ["## Fire Overview"]
    
    # Add current date (MM-DD format)
    analysis_mmdd = ctx.summary.get("analysis_mmdd")
    analysis_date = ctx.summary.get("analysis_date")
    
    if analysis_mmdd:
        lines.append(f"- Current date: {analysis_mmdd}")
    elif analysis_date:
        # Format date to MM-DD (remove year) as fallback
        try:
            from datetime import datetime
            parsed_date = datetime.strptime(analysis_date, "%Y-%m-%d")
            formatted_date = parsed_date.strftime("%m-%d")
            lines.append(f"- Current date: {formatted_date}")
        except (ValueError, TypeError):
            # Fallback to original if parsing fails
            lines.append(f"- Current date: {analysis_date}")
    
    # Get fire overview from analysis.py structure
    fire_overview_data = ctx.summary.get("fire_overview", {})
    weather_conditions = ctx.summary.get("weather_conditions", {})
    no_fire_points_today = ctx.summary.get("no_fire_points_today", False)
    
    if no_fire_points_today:
        consecutive_days = ctx.summary.get("no_fire_consecutive_days", 1)
        if consecutive_days == 1:
            lines.append("- No active FIRMS hotspots detected today (1st consecutive day). This could indicate data gaps or actual fire suppression. Use previous-day context and global conditions to provide TODAY's personnel and daily budget (NOT cumulative).")
        else:
            lines.append(f"- No active FIRMS hotspots detected today ({consecutive_days} consecutive days). Consider whether this represents true fire suppression or potential data collection issues. Use previous-day context and global conditions to provide TODAY's personnel and daily budget (NOT cumulative).")
    
    if fire_overview_data:
        if has_previous:
            # Display comparison format
            lines.extend(_build_fire_overview_comparison(ctx, fire_overview_data))
        else:
            # Display simple format
            lines.extend([
                f"- Total clusters: {fire_overview_data.get('num_clusters', 0)}",
                f"- Total fire points: {fire_overview_data.get('total_fire_points', 0)}",
                f"- Total FRP: {fire_overview_data.get('total_frp', 0):.1f} MW",
                f"- Total area: {format_area(fire_overview_data.get('total_area_m2', 0), precision=2)}",
                f"- Max cluster FRP: {fire_overview_data.get('max_frp', 0):.1f} MW",
                f"- Max brightness: {fire_overview_data.get('max_brightness', 0):.1f} K"
            ])
    
    # Include weather conditions only if environmental info is enabled
    if weather_conditions and _should_include_environmental_info(ctx):
        weather_parts = []
        if weather_conditions.get('bi') is not None:
            weather_parts.append(f"BI={weather_conditions['bi']}")
        if weather_conditions.get('tmmx') is not None:
            tmax_display = format_temperature(weather_conditions['tmmx'], precision=1)
            weather_parts.append(f"Tmax={tmax_display}")
        if weather_conditions.get('tmmn') is not None:
            tmin_display = format_temperature(weather_conditions['tmmn'], precision=1)
            weather_parts.append(f"Tmin={tmin_display}")
        if weather_conditions.get('vs') is not None:
            weather_parts.append(f"Wind={weather_conditions['vs']} m/s")
        if weather_conditions.get('fm1') is not None:
            weather_parts.append(f"FM1={weather_conditions['fm1']:.1f}%")
        
        if weather_parts:
            lines.append(f"- Weather conditions (fire-point weighted): {', '.join(weather_parts)}")
    
    return "\n".join(lines)


def affected_areas(ctx: Context) -> str:
    """Build affected areas section with optional comparison"""
    affected = ctx.summary.get("affected_areas", {})
    if not affected:
        return ""
    
    # Check if we have previous data for comparison
    has_previous = bool(ctx.previous_summary)
    
    if has_previous:
        lines = ["## Affected Areas vs Yesterday"]
    else:
        lines = ["## Affected Areas"]
    
    # Counties information
    counties = affected.get("counties", [])
    fire_stations = ctx.summary.get("fire_stations", {})
    
    if has_previous:
        # Display comparison format
        lines.extend(_build_affected_areas_comparison(ctx, affected, fire_stations))
    else:
        # Display simple format - only include counties if environmental info is enabled
        if counties and _should_include_environmental_info(ctx):
            lines.append(f"- Counties affected: {', '.join(counties)} ({len(counties)} total)")
        
        # Population information - only include if environmental info is enabled
        if _should_include_environmental_info(ctx):
            total_pop = affected.get("total_population_affected", 0)
            if total_pop > 0:
                lines.append(f"- Total population affected: {total_pop:.0f}")
        
        # Fire stations information - only include if environmental info is enabled
        if _should_include_environmental_info(ctx):
            fire_stations_num = fire_stations.get("total_stations", 0)
            if fire_stations_num > 0:
                lines.append(f"- Fire stations in area: {fire_stations_num}")
                
                # Optionally add nearest station distance - use consistent unit formatting
                nearest_distance_m = fire_stations.get("nearest_distance_m")
                if nearest_distance_m is not None:
                    lines.append(f"- Nearest fire station: {format_distance(float(nearest_distance_m), precision=1)} away")
    
    return "\n".join(lines) if len(lines) > 1 else ""


def previous_context(ctx: Context) -> str:
    """Build previous analysis context for incremental mode"""
    if not ctx.previous_analysis:
        return ""
    
    # Extract previous values
    prev_resources = ctx.previous_analysis.get("resource_requirements", {})
    prev_personnel = prev_resources.get("daily_personnel", {}).get("value", "N/A")
    prev_budget = prev_resources.get("daily_budget", {}).get("value", "N/A")
    
    # Extract previous reasoning
    prev_reasoning = ctx.previous_analysis.get("analysis_reasoning", {})
    prev_overall = prev_reasoning.get("overall_reasoning", "N/A")
    
    # Extract cumulative cost information
    cumulative_info = ctx.previous_analysis.get("cumulative_cost_info", {})
    total_cumulative = cumulative_info.get("total_cumulative_cost", "N/A")
    prev_day_cumulative = cumulative_info.get("previous_day_cumulative", "N/A")
    
    budget_str = f"${prev_budget:.0f}" if isinstance(prev_budget, (int, float)) else str(prev_budget)
    
    # Build cumulative cost display
    cumulative_lines = []
    if isinstance(total_cumulative, (int, float)) and isinstance(prev_day_cumulative, (int, float)):
        cumulative_str = f"${total_cumulative:.0f}"
        prev_cumulative_str = f"${prev_day_cumulative:.0f}"
        increase = total_cumulative - prev_day_cumulative
        increase_str = f"${increase:.0f}" if increase >= 0 else f"-${abs(increase):.0f}"
        cumulative_lines.append(f"- Total cumulative cost: {cumulative_str} (previous day: {prev_cumulative_str}, increase: {increase_str})")
    elif isinstance(total_cumulative, (int, float)):
        cumulative_str = f"${total_cumulative:.0f}"
        cumulative_lines.append(f"- Total cumulative cost: {cumulative_str}")
    
    # Combine all context information
    context_lines = [
        "## Previous Analysis Context",
        f"- Previous personnel: {prev_personnel} people",
        f"- Previous daily budget: {budget_str}",
    ]
    
    if cumulative_lines:
        context_lines.extend(cumulative_lines)
    
    context_lines.append(f"- Previous reasoning: {prev_overall}")
    
    return "\n".join(context_lines)


def cumulative_context(ctx: Context) -> str:
    """Build cumulative context section with rolling statistics and trends"""
    from config import CUMULATIVE_CONFIG
    
    if not CUMULATIVE_CONFIG.get("enabled", True):
        return ""
    
    if not ctx.previous_analysis:
        return ""
    
    # Extract cumulative data from previous_analysis
    cumulative_info = ctx.previous_analysis.get("cumulative_context_info", {})
    if not cumulative_info:
        return ""
    
    lines = ["## Cumulative Context"]
    
    # Basic cumulative metrics
    total_cost = cumulative_info.get("total_cumulative_cost", 0)
    total_personnel_days = cumulative_info.get("total_cumulative_personnel_days", 0)
    days_since_start = cumulative_info.get("days_since_fire_start", 0)
    
    cost_precision = CUMULATIVE_CONFIG.get("precision", {}).get("cost", 0)
    personnel_precision = CUMULATIVE_CONFIG.get("precision", {}).get("personnel", 0)
    days_precision = CUMULATIVE_CONFIG.get("precision", {}).get("days", 0)
    
    lines.append(f"- Total cumulative cost: ${total_cost:,.{cost_precision}f}")
    lines.append(f"- Total cumulative personnel-days: {total_personnel_days:,.{personnel_precision}f}")
    lines.append(f"- Days since fire start: {days_since_start:.{days_precision}f}")
    
    # Rolling averages
    rolling_windows = CUMULATIVE_CONFIG.get("rolling_windows", [3, 7])
    rolling_stats = cumulative_info.get("rolling_stats", {})
    
    for window in rolling_windows:
        window_key = f"{window}day"
        if window_key in rolling_stats:
            stats = rolling_stats[window_key]
            avg_cost = stats.get("avg_daily_cost", 0)
            avg_personnel = stats.get("avg_daily_personnel", 0)
            
            lines.append(f"- {window}-day rolling avg daily cost: ${avg_cost:,.{cost_precision}f}")
            lines.append(f"- {window}-day rolling avg daily personnel: {avg_personnel:,.{personnel_precision}f}")
    
    # Resource intensity trends (if available)
    trends = cumulative_info.get("trends", {})
    if trends:
        cost_trend = trends.get("cost_trend", "stable")
        personnel_trend = trends.get("personnel_trend", "stable")
        lines.append(f"- Recent cost trend: {cost_trend}")
        lines.append(f"- Recent personnel trend: {personnel_trend}")
    
    # Fire intensity rolling metrics (if available)
    # Rolling metrics come from previous_summary as they are calculated from historical data
    fire_overview = ctx.previous_summary.get("fire_overview", {}) if ctx.previous_summary else {}
    rolling_fire = fire_overview.get("rolling", {})
    if rolling_fire:
        lines.append("")  # Add separator
        lines.append("## Fire Intensity Rolling Metrics")
        
        # Get days info to control 7-day metrics display
        days_since_start = cumulative_info.get("days_since_fire_start", 0)
        show_7day = days_since_start >= 7
        
        # Fire points metrics
        fire_points_data = rolling_fire.get("fire_points", {})
        if fire_points_data:
            lines.append(f"- 3-day avg fire points: {fire_points_data.get('3day_avg_total', 'N/A')}")
            lines.append(f"- 3-day max fire points: {fire_points_data.get('3day_max_total', 'N/A')}")
            
            # 7-day metrics (only shown after day 7)
            if show_7day:
                lines.append(f"- 7-day avg fire points: {fire_points_data.get('7day_avg_total', 'N/A')}")
                lines.append(f"- 7-day max fire points: {fire_points_data.get('7day_max_total', 'N/A')}")
            
            # Convert ratio to percentage
            ratio_val = fire_points_data.get('current_vs_max_ratio', 'N/A')
            if ratio_val != 'N/A' and ratio_val is not None:
                try:
                    ratio_pct = f"{float(ratio_val) * 100:.1f}%"
                except (ValueError, TypeError):
                    ratio_pct = 'N/A'
            else:
                ratio_pct = 'N/A'
            lines.append(f"- Current fire points vs historical max: {ratio_pct}")
            
            # Global max information
            global_max_fire_points = fire_points_data.get('global_max_value', 'N/A')
            days_since_max_fire_points = fire_points_data.get('days_since_global_max', 'N/A')
            if global_max_fire_points != 'N/A' and days_since_max_fire_points != 'N/A':
                lines.append(f"- Global max fire points: {global_max_fire_points} ({days_since_max_fire_points})")
        
        # Area metrics (with unit conversion)
        area_data = rolling_fire.get("area", {})
        if area_data:
            from utils.unit_converter import format_area
            
            # 3-day metrics
            avg_3d = area_data.get('3day_avg_total', 'N/A')
            max_3d = area_data.get('3day_max_total', 'N/A')
            
            if avg_3d != 'N/A':
                try:
                    avg_3d_formatted = format_area(float(avg_3d), precision=2)
                except (ValueError, TypeError):
                    avg_3d_formatted = 'N/A'
            else:
                avg_3d_formatted = 'N/A'
                
            if max_3d != 'N/A':
                try:
                    max_3d_formatted = format_area(float(max_3d), precision=2)
                except (ValueError, TypeError):
                    max_3d_formatted = 'N/A'
            else:
                max_3d_formatted = 'N/A'
            
            lines.append(f"- 3-day avg total area: {avg_3d_formatted}")
            lines.append(f"- 3-day max total area: {max_3d_formatted}")
            
            # 7-day metrics (only shown after day 7)
            if show_7day:
                avg_7d = area_data.get('7day_avg_total', 'N/A')
                max_7d = area_data.get('7day_max_total', 'N/A')
                
                if avg_7d != 'N/A':
                    try:
                        avg_7d_formatted = format_area(float(avg_7d), precision=2)
                    except (ValueError, TypeError):
                        avg_7d_formatted = 'N/A'
                else:
                    avg_7d_formatted = 'N/A'
                    
                if max_7d != 'N/A':
                    try:
                        max_7d_formatted = format_area(float(max_7d), precision=2)
                    except (ValueError, TypeError):
                        max_7d_formatted = 'N/A'
                else:
                    max_7d_formatted = 'N/A'
                
                lines.append(f"- 7-day avg total area: {avg_7d_formatted}")
                lines.append(f"- 7-day max total area: {max_7d_formatted}")
            
            # Convert ratio to percentage
            area_ratio_val = area_data.get('current_vs_max_ratio', 'N/A')
            if area_ratio_val != 'N/A' and area_ratio_val is not None:
                try:
                    area_ratio_pct = f"{float(area_ratio_val) * 100:.1f}%"
                except (ValueError, TypeError):
                    area_ratio_pct = 'N/A'
            else:
                area_ratio_pct = 'N/A'
            lines.append(f"- Current area vs historical max: {area_ratio_pct}")
            
            # Global max information (with unit conversion)
            global_max_area = area_data.get('global_max_value', 'N/A')
            days_since_max_area = area_data.get('days_since_global_max', 'N/A')
            if global_max_area != 'N/A' and days_since_max_area != 'N/A':
                try:
                    global_max_area_formatted = format_area(float(global_max_area), precision=2)
                except (ValueError, TypeError):
                    global_max_area_formatted = 'N/A'
                
                if global_max_area_formatted != 'N/A':
                    lines.append(f"- Global max area: {global_max_area_formatted} ({days_since_max_area})")
    
    return "\n".join(lines)


def cluster_highlights(ctx: Context) -> str:
    """Build cluster highlights section - show main clusters with full info, count others"""
    lines = ["## Cluster Highlights"]
    
    # Handle empty clusters for no fire points days
    if not ctx.clusters_data:
        no_fire_points_today = ctx.summary.get("no_fire_points_today", False)
        if no_fire_points_today:
            lines.append("- No clusters detected today (no active fire points)")
            return "\n".join(lines)
        else:
            return "\n".join(lines)
    
    # Sort clusters by fire points (descending)
    sorted_clusters = sorted(ctx.clusters_data, 
                           key=lambda c: c.get("cluster_info", {}).get("points", 0), 
                           reverse=True)
    
    if not sorted_clusters:
        return "\n".join(lines)
    
    # Find main clusters (largest + within 10% threshold)
    max_points = sorted_clusters[0].get("cluster_info", {}).get("points", 0)
    threshold = max_points * 0.9  # 90% of max points
    
    main_clusters = []
    other_clusters = []
    
    for cluster in sorted_clusters:
        points = cluster.get("cluster_info", {}).get("points", 0)
        if points >= threshold:
            main_clusters.append(cluster)
        else:
            other_clusters.append(cluster)
    
    # Format main clusters with full information
    for cluster in main_clusters:
        cluster_line = _build_cluster_line(cluster, ctx)
        lines.append(cluster_line)
    
    # Add count of other clusters if any
    if other_clusters:
        lines.append(f"- Other clusters: {len(other_clusters)} smaller clusters not detailed.")
    
    return "\n".join(lines)


def cluster_details(ctx: Context) -> str:
    """Build cluster details section with unit formatting"""
    lines = ["## Cluster Details"]
    
    # Handle empty clusters for no fire points days
    if not ctx.clusters_data:
        no_fire_points_today = ctx.summary.get("no_fire_points_today", False)
        if no_fire_points_today:
            lines.append("- No clusters detected today (no active fire points)")
            return "\n".join(lines)
        else:
            return "\n".join(lines)
    
    for cluster in ctx.clusters_data:
        cluster_line = _build_cluster_line(cluster, ctx)
        lines.append(cluster_line)
    
    return "\n".join(lines)

def global_change_hints(ctx: Context) -> str:
    """Build global change hints section for fire analysis"""
    
    hints = ctx.summary.get("global_change_hints", {})
    if not hints:
        return ""
    
    lines = ["## Change Summary"]
    
    for key, value in hints.items():
        if isinstance(value, str) and ('↑' in value or '↓' in value or 'change' in value.lower()):
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        elif isinstance(value, (int, float)):
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(lines) if len(lines) > 1 else ""


def rag_context(ctx: Context) -> str:
    """Build RAG context section - directly embed formatted RAG context"""
    if not ctx.rag_context or not ctx.rag_context.strip():
        return ""
    
    # Direct embedding - trust the RAG provider's formatting
    return f"""{ctx.rag_context}"""


def _build_fire_overview_comparison(ctx: Context, fire_overview_data: Dict[str, Any]) -> List[str]:
    """Build fire overview comparison lines"""
    # Import config here to avoid circular import
    from config import COMPARISON_CONFIG
    
    # Get configuration
    use_arrows = COMPARISON_CONFIG.get("use_arrows", True)
    show_percent = COMPARISON_CONFIG.get("show_percent_change", True)
    percent_min_change = COMPARISON_CONFIG.get("percent_min_change", 0.05)
    float_precision = COMPARISON_CONFIG.get("float_precision", 1)
    pct_precision = COMPARISON_CONFIG.get("percent_precision", 1)
    no_fire_policy = COMPARISON_CONFIG.get("no_fire_policy", "drop_to_zero_with_notice")
    
    lines = []
    curr_no_fire = ctx.summary.get("no_fire_points_today", False)
    prev_fire_overview = ctx.previous_summary.get("fire_overview", {})
    prev_no_fire = (prev_fire_overview.get("total_fire_points", 0) == 0)
    
    if curr_no_fire and not prev_no_fire and no_fire_policy == "drop_to_zero_with_notice":
        lines.append("- Note: No active FIRMS hotspots today; fire-overview metrics drop to zero by definition.")
    
    # Compare fire overview metrics
    fire_metrics = COMPARISON_CONFIG.get("metrics", {}).get("fire_overview", [
        "total_fire_points", "num_clusters", "total_frp", "total_area_m2", "max_frp", "max_brightness"
    ])
    
    for metric in fire_metrics:
        prev_val = prev_fire_overview.get(metric, 0)
        curr_val = fire_overview_data.get(metric, 0)
        
        if metric in ["total_fire_points", "num_clusters"]:
            # Integer metrics
            comparison = format_value_with_arrow(prev_val, curr_val, is_int=True)
            arrow = comparison['arrow'] if use_arrows else ""
            delta = comparison['delta']
            metric_name = metric.replace("_", " ").title()
            lines.append(f"- {metric_name}: {curr_val} ({arrow} {delta})")
        
        elif metric == "total_area_m2":
            # Area metric with unit formatting
            comparison = format_value_with_arrow(prev_val, curr_val, is_int=False, 
                                                percent_threshold=percent_min_change)
            arrow = comparison['arrow'] if use_arrows else ""
            # Use centralized delta calculation with proper unit conversion
            delta = format_delta_with_units(prev_val, curr_val, 'area', 'm2', 
                                          float_precision, pct_precision, show_percent)
            curr_display = format_area(curr_val, precision=2)
            lines.append(f"- Total area: {curr_display} ({arrow} {delta})")
        
        else:
            # Float metrics (FRP, brightness)
            comparison = format_value_with_arrow(prev_val, curr_val, is_int=False,
                                                percent_threshold=percent_min_change,
                                                precision=float_precision,
                                                pct_precision=pct_precision,
                                                show_percent=show_percent)
            arrow = comparison['arrow'] if use_arrows else ""
            delta = comparison['delta']
            
            if "frp" in metric.lower():
                lines.append(f"- {metric.replace('_', ' ').title()}: {curr_val:.1f} MW ({arrow} {delta})")
            elif "brightness" in metric.lower():
                # Brightness is in fixed Kelvin units, no conversion needed
                lines.append(f"- {metric.replace('_', ' ').title()}: {curr_val:.1f} K ({arrow} {delta})")
            else:
                lines.append(f"- {metric.replace('_', ' ').title()}: {curr_val:.1f} ({arrow} {delta})")
    
    return lines


def _build_affected_areas_comparison(ctx: Context, affected: Dict[str, Any], fire_stations: Dict[str, Any]) -> List[str]:
    """Build affected areas comparison lines"""
    # Import config here to avoid circular import
    from config import COMPARISON_CONFIG
    
    # Get configuration
    use_arrows = COMPARISON_CONFIG.get("use_arrows", True)
    show_percent = COMPARISON_CONFIG.get("show_percent_change", True)
    percent_min_change = COMPARISON_CONFIG.get("percent_min_change", 0.05)
    float_precision = COMPARISON_CONFIG.get("float_precision", 1)
    pct_precision = COMPARISON_CONFIG.get("percent_precision", 1)
    show_stations_no_fire = COMPARISON_CONFIG.get("show_fire_stations_when_no_fire", False)
    
    lines = []
    curr_no_fire = ctx.summary.get("no_fire_points_today", False)
    prev_affected_areas = ctx.previous_summary.get("affected_areas", {})
    prev_fire_stations = ctx.previous_summary.get("fire_stations", {})
    
    # Compare affected areas metrics
    affected_metrics = COMPARISON_CONFIG.get("metrics", {}).get("affected_areas", [
        "counties", "total_population_affected", "num_counties"
    ])
    
    for metric in affected_metrics:
        if metric == "counties":
            # Special handling for counties list - only include if environmental info is enabled
            if _should_include_environmental_info(ctx):
                prev_counties = prev_affected_areas.get("counties", [])
                curr_counties = affected.get("counties", [])
                county_changes = compare_lists(prev_counties, curr_counties)
                
                if curr_no_fire:
                    if prev_counties:
                        removed_str = "{" + ", ".join(county_changes["removed"]) + "}" if county_changes["removed"] else ""
                        lines.append(f"- Counties: removed {removed_str}, now none")
                    else:
                        lines.append("- Counties: unchanged (none)")
                else:
                    if county_changes["added"] or county_changes["removed"]:
                        added_str = "{" + ", ".join(county_changes["added"]) + "}" if county_changes["added"] else ""
                        removed_str = "{" + ", ".join(county_changes["removed"]) + "}" if county_changes["removed"] else ""
                        changes = []
                        if added_str:
                            changes.append(f"added {added_str}")
                        if removed_str:
                            changes.append(f"removed {removed_str}")
                        lines.append(f"- Counties: {', '.join(changes)}; now {county_changes['total_now']}")
                    else:
                        lines.append(f"- Counties: unchanged ({county_changes['total_now']})")
        
        else:
            # Numeric metrics - only include population if environmental info is enabled
            if "population" in metric and not _should_include_environmental_info(ctx):
                continue
                
            prev_val = prev_affected_areas.get(metric, 0)
            curr_val = affected.get(metric, 0)
            
            comparison = format_value_with_arrow(prev_val, curr_val, is_int=True)
            arrow = comparison['arrow'] if use_arrows else ""
            delta = comparison['delta']
            metric_name = metric.replace("_", " ").title()
            
            if "population" in metric:
                lines.append(f"- {metric_name}: {curr_val:.0f} ({arrow} {delta})")
            else:
                lines.append(f"- {metric_name}: {curr_val} ({arrow} {delta})")
    
    # Fire stations comparison - only include if environmental info is enabled
    if _should_include_environmental_info(ctx):
        fire_station_metrics = COMPARISON_CONFIG.get("metrics", {}).get("fire_stations", [
            "total_stations", "nearest_distance_m"
        ])
        
        for metric in fire_station_metrics:
            prev_val = prev_fire_stations.get(metric, 0)
            curr_val = fire_stations.get(metric, 0)
            
            if curr_no_fire and not show_stations_no_fire:
                if metric == "total_stations":
                    lines.append(f"- Fire stations: N/A today (vs {prev_val})")
                elif metric == "nearest_distance_m" and prev_val > 0:
                    prev_display = format_distance(float(prev_val), precision=1)
                    lines.append(f"- Nearest station: N/A today (vs {prev_display})")
            else:
                if metric == "total_stations":
                    comparison = format_value_with_arrow(prev_val, curr_val, is_int=True)
                    arrow = comparison['arrow'] if use_arrows else ""
                    delta = comparison['delta']
                    lines.append(f"- Fire stations in area: {curr_val} ({arrow} {delta})")
                
                elif metric == "nearest_distance_m":
                    if prev_val > 0 and curr_val > 0:
                        comparison = format_value_with_arrow(prev_val, curr_val, is_int=False,
                                                            percent_threshold=percent_min_change)
                        arrow = comparison['arrow'] if use_arrows else ""
                        # Use centralized delta calculation with proper unit conversion
                        delta = format_delta_with_units(prev_val, curr_val, 'distance', 'm', 
                                                      float_precision, pct_precision, show_percent)
                        curr_display = format_distance(float(curr_val), precision=1)
                        lines.append(f"- Nearest station: {curr_display} ({arrow} {delta})")
    
    return lines


# daily_comparison function has been removed - functionality merged into fire_overview and affected_areas


def instruction_recall(ctx: Context) -> str:
    """Build instruction recall section - system prompt recall for user prompt"""
    lines = [
        "## Instruction",
        "- Compare today's situation with yesterday clearly.",
        # "- Pay attention to rolling metrics, and avoid abrupt drops in personnel and budget.",
        "- Weigh trade-offs: FRP/area decline vs. more clusters and global max indicators.",
        "- Keep personnel and budget within RAG-guided bounds unless strong evidence says otherwise.",
        "- Output strictly follows the schema."
    ]
    return "\n".join(lines)


# =============================================================================
# Section Registry
# =============================================================================

# Registry of all available section builders
SECTION_BUILDERS: Dict[str, SectionBuilder] = {
    # User prompt sections
    "fire_overview": fire_overview,
    "affected_areas": affected_areas,
    "previous_context": previous_context,
    "cumulative_context": cumulative_context,
    "rag_context": rag_context,
    "cluster_highlights": cluster_highlights,
    "cluster_details": cluster_details,
    "clusters": None,  # Dynamic section, handled specially in build_user_prompt_sections
    "global_change_hints": global_change_hints,
    "instruction_recall": instruction_recall,
}


# =============================================================================
# Section Enablement Logic
# =============================================================================

def is_section_enabled(section_name: str, ctx: Context, config_toggles: Dict[str, Any] = None) -> bool:
    """Determine if a section should be enabled based on context and configuration"""
    
    config_toggles = config_toggles or {}
    
    # Check explicit configuration first
    if section_name in config_toggles:
        toggle_value = config_toggles[section_name]
        if isinstance(toggle_value, bool):
            return toggle_value
        elif toggle_value == "auto":
            pass  # Fall through to auto logic
        else:
            return bool(toggle_value)
    
    # Auto-enablement logic based on context
    if section_name == "affected_areas":
        return bool(ctx.summary.get("affected_areas"))
    
    elif section_name == "previous_context":
        return bool(ctx.previous_analysis)
    
    elif section_name == "cumulative_context":
        from config import CUMULATIVE_CONFIG
        return (CUMULATIVE_CONFIG.get("enabled", True) and 
                bool(ctx.previous_analysis) and 
                bool(ctx.previous_analysis.get("cumulative_context_info")))
    
    
    elif section_name == "global_change_hints":
        return bool(ctx.summary.get("global_change_hints"))
    
    elif section_name == "rag_context":
        return bool(ctx.rag_context and ctx.rag_context.strip())
    
    # Default sections are always enabled
    elif section_name in ["fire_overview", "cluster_highlights", "cluster_details", "clusters", "instruction_recall"]:
        return True
    
    # Unknown sections are disabled by default
    return False


def build_user_prompt_sections(ctx: Context, section_order: List[str], config_toggles: Dict[str, Any] = None) -> List[str]:
    """Build user prompt sections based on order and enablement"""
    
    sections = []
    config_toggles = config_toggles or {}
    
    for section_name in section_order:
        # Remove optional marker if present
        clean_name = section_name.rstrip('?')
        
        # Handle dynamic "clusters" section
        if clean_name == "clusters":
            # Map to actual section based on cluster_display_mode
            cluster_mode = config_toggles.get("cluster_display_mode", "details")
            if cluster_mode == "highlights":
                actual_section = "cluster_highlights"
            else:  # default to "details"
                actual_section = "cluster_details"
            
            # Use the selected section
            if is_section_enabled(actual_section, ctx, config_toggles):
                if actual_section in SECTION_BUILDERS:
                    try:
                        section_content = SECTION_BUILDERS[actual_section](ctx)
                        if section_content.strip():
                            sections.append(section_content)
                    except Exception as e:
                        print(f"Warning: Failed to build section '{actual_section}': {e}")
            continue
        
        # Check if section is enabled
        if is_section_enabled(clean_name, ctx, config_toggles):
            if clean_name in SECTION_BUILDERS:
                try:
                    section_content = SECTION_BUILDERS[clean_name](ctx)
                    if section_content.strip():  # Only add non-empty sections
                        sections.append(section_content)
                except Exception as e:
                    print(f"Warning: Failed to build section '{clean_name}': {e}")
    
    return sections


# =============================================================================
# Comparison Helper Functions (migrated to utils/comparison_utils.py)
# =============================================================================
# Note: format_arrow, format_delta, arrow_for_int, arrow_for_float, fmt_delta_int, fmt_delta_float
# have been replaced by format_value_with_arrow() from utils/comparison_utils.py.
# compare_counties has been replaced by compare_lists() from utils/comparison_utils.py.

# =============================================================================
# Helper Functions
# =============================================================================

def _format_terrain_description(cluster: Dict[str, Any], ctx: Context) -> str:
    """Format terrain description - choose between detailed and summary versions"""
    
    # Import config here to get latest values
    from config import PROMPT_SECTION_CONFIG
    
    # Get format from summary override first, then from config
    terrain_format = ctx.summary.get("terrain_format")
    if not terrain_format:
        # Get from current template
        current_template = PROMPT_SECTION_CONFIG.get("current_template", "template1")
        template_config = PROMPT_SECTION_CONFIG.get("templates", {}).get(current_template, {})
        fire_analysis_config = template_config.get("fire_analysis", {})
        user_config = fire_analysis_config.get("user", {})
        terrain_format = user_config.get("toggles", {}).get("terrain_format", "detailed")
    
    use_summary_format = terrain_format == "summary"
    
    if use_summary_format:
        # Use summary statistics format
        terrain_conditions = ctx.summary.get("terrain_conditions", {})
        if terrain_conditions:
            parts = []
            
            # Risk assessment
            high_risk = terrain_conditions.get("avg_high_risk_percent", 0)
            overall_risk = terrain_conditions.get("avg_overall_risk_score", 0)
            if high_risk > 0:
                parts.append(f"high-risk vegetation {high_risk}%")
            
            # Fuel continuity
            fuel_continuity = terrain_conditions.get("avg_continuous_fuel_percent", 0)
            if fuel_continuity > 0:
                parts.append(f"fuel continuity {fuel_continuity}%")
            
            # Diversity
            diversity = terrain_conditions.get("avg_diversity_index", 0)
            land_types = terrain_conditions.get("avg_num_land_types", 0)
            if diversity > 0:
                parts.append(f"diversity index {diversity:.2f}")
            if land_types > 0:
                parts.append(f"{int(land_types)} land types")
            
            # Overall risk score
            if overall_risk > 0:
                risk_level = "Very High" if overall_risk >= 4.5 else "High" if overall_risk >= 3.5 else "Moderate"
                parts.append(f"risk level {risk_level}")
            
            if parts:
                return f"[{', '.join(parts)}]"
    
    # Default to detailed format
    terrain_description = cluster.get("terrain_analysis", "Unknown terrain")
    if isinstance(terrain_description, dict):
        terrain_description = terrain_description.get("natural_description", "Unknown terrain")
    
    return terrain_description
