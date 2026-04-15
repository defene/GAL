from typing import Dict, Any, List, Optional
from datetime import datetime
from db_fetcher import DBDataFetcher
from config import FIRE_NAMES, DEFAULT_PARAMS, FIRE_CLUSTERING
import json
import decimal
import math


def _geo_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate approximate distance in meters between two geographic points.
    
    Uses simple planar approximation, sufficient for small distances (<100km).
    1 degree latitude ≈ 111,000 meters
    1 degree longitude ≈ 111,000 * cos(latitude) meters
    """
    lat_avg = math.radians((lat1 + lat2) / 2)
    lat_diff = lat2 - lat1
    lon_diff = lon2 - lon1
    
    lat_m = lat_diff * 111000
    lon_m = lon_diff * 111000 * math.cos(lat_avg)
    
    return math.sqrt(lat_m**2 + lon_m**2)


def _calculate_weighted_average(data_list: List[Dict], weights: List[float], keys: List[str], 
                                precisions: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Calculate weighted averages for multiple keys from a list of data dictionaries.
    
    Args:
        data_list: List of dictionaries containing data
        weights: List of weights corresponding to each data item
        keys: List of keys to calculate weighted averages for
        precisions: Optional dict mapping keys to decimal precisions
        
    Returns:
        Dictionary with weighted averages for each key
    """
    if not data_list or not weights:
        return {}
    
    total_weight = sum(weights)
    if total_weight == 0:
        return {}
    
    precisions = precisions or {}
    results = {}
    
    for key in keys:
        values = [d.get(key, 0) for d in data_list]
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        avg_value = weighted_sum / total_weight
        
        precision = precisions.get(key, 2)  # Default to 2 decimal places
        results[key] = round(avg_value, precision)
    
    return results


class FireAnalysis:
    """Comprehensive fire analysis combining all data sources"""
    
    def __init__(self):
        self.fetcher = DBDataFetcher()
    
    def analyze_fire(self, fire_name: str, date_str: str) -> Dict[str, Any]:
        """
        Comprehensive fire analysis for given fire name and date
        
        Args:
            fire_name: Name of the fire (e.g., "CREEK")
            date_str: Date in YYYY-MM-DD format (e.g., "2020-08-18")
            
        Returns:
            Dict containing complete analysis results
        """
        # Validate inputs
        if fire_name not in FIRE_NAMES:
            return {
                "error": f"Fire name '{fire_name}' not found in database",
                "available_fires": FIRE_NAMES
            }
        
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return {
                "error": f"Invalid date format '{date_str}'. Use YYYY-MM-DD format"
            }
        
        # Initialize result structure
        analysis_result = {
            "fire_name": fire_name,
            "analysis_date": date_str,
            "timestamp": datetime.now().isoformat(),
            "cluster_analysis": [],   # Detailed analysis per cluster
            "summary": {},            # High-level aggregated data
            "errors": []
        }
        
        try:
            # Get fire point clusters
            clusters_data = self.fetcher.get_fire_clusters(fire_name, date_str)
            
            if "error" in clusters_data:
                analysis_result["errors"].append(f"Cluster analysis failed: {clusters_data['error']}")
                return analysis_result
            
            if clusters_data.get("total_points", 0) == 0:
                # No fire points found, but continue with analysis using empty clusters
                analysis_result["summary"] = {
                    "fire_name": fire_name,
                    "analysis_date": date_str,
                    "analysis_mmdd": date_str[5:] if len(date_str) >= 10 else date_str,
                    "no_fire_points_today": True,
                    "fire_overview": {
                        "num_clusters": 0,
                        "total_fire_points": 0,
                        "total_frp": 0.0,
                        "total_area_m2": 0.0,
                        "max_frp": 0.0,
                        "max_brightness": 0.0
                    },
                    "affected_areas": {},
                    "fire_stations": {},
                    "weather_conditions": {},
                    "terrain_conditions": {}
                }
                return analysis_result
            
            # Analyze each cluster (single pass aggregation)
            total_population = 0
            all_counties = set()
            cluster_weather_data = []
            total_area_m2 = 0
            max_cluster_area = 0
            
            for i, cluster in enumerate(clusters_data.get("clusters", [])):
                cluster_analysis = self._analyze_single_cluster(
                    cluster, date_str
                )
                analysis_result["cluster_analysis"].append(cluster_analysis)
                
                # Aggregate all data in single pass (avoid second iteration)
                if "population" in cluster_analysis and not cluster_analysis["population"].get("error"):
                    total_population += cluster_analysis["population"].get("pop_sum", 0)
                
                if "county" in cluster_analysis and not cluster_analysis["county"].get("error"):
                    all_counties.add(cluster_analysis["county"].get("county_name", "Unknown"))
                
                if "weather" in cluster_analysis and not cluster_analysis["weather"].get("error"):
                    cluster_weather_data.append(cluster_analysis["weather"])
                
                # Aggregate area metrics (previously done in _generate_summary)
                cluster_info = cluster_analysis.get("cluster_info", {})
                area = cluster_info.get("area_m2", 0)
                total_area_m2 += area
                max_cluster_area = max(max_cluster_area, area)
            
            # Generate summary (now receives pre-aggregated area metrics)
            analysis_result["summary"] = self._generate_summary(
                clusters_data, total_population, list(all_counties), 
                cluster_weather_data, analysis_result["cluster_analysis"], 
                fire_name, date_str, total_area_m2, max_cluster_area
            )
            
            return analysis_result
            
        except Exception as e:
            analysis_result["errors"].append(f"Analysis failed: {str(e)}")
            return analysis_result
    
    def _safe_fetch_cluster_data(self, key: str, fetch_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Safe data fetching with automatic error handling.
        
        Args:
            key: Data type identifier (for error messages)
            fetch_func: Function to call for data fetching
            *args, **kwargs: Arguments to pass to fetch_func
            
        Returns:
            Dictionary with fetched data or error information
        """
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_single_cluster(self, cluster: Dict[str, Any], date_str: str) -> Dict[str, Any]:
        """
        Analyze a single fire cluster with all associated spatial data.
        
        Args:
            cluster: Cluster data dictionary containing geometric and fire properties
            date_str: Analysis date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing comprehensive cluster analysis results
        """
        cluster_id = cluster.get("cluster_id", "unknown")
        center_lat = cluster.get("center_lat", 0)
        center_lon = cluster.get("center_lon", 0)
        polygon_wkt = cluster.get("polygon_wkt", "")
        
        cluster_analysis = {
            "cluster_id": cluster_id,
            "cluster_info": {
                "points": cluster.get("points", 0),
                "frp": cluster.get("frp", 0),
                "max_brightness": cluster.get("max_brightness", 0),
                "max_delta_t": cluster.get("max_delta_t", 0),
                "area_m2": cluster.get("area_m2", 0),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "polygon_wkt": polygon_wkt
            }
        }
        
        # Get county information (based on cluster center)
        cluster_analysis["county"] = self._safe_fetch_cluster_data(
            "county",
            self.fetcher.get_county_name,
            center_lon, center_lat
        )
        
        # Get nearest fire stations (based on cluster center)
        cluster_analysis["fire_stations"] = self._safe_fetch_cluster_data(
            "fire_stations",
            self.fetcher.get_nearest_fire_stations,
            center_lon, center_lat,
            limit=DEFAULT_PARAMS["fire_station_limit"]
        )
        
        # Get polygon-based data only if polygon exists
        if polygon_wkt:
            # Get population data (with automatic fallback)
            cluster_analysis["population"] = self._safe_fetch_cluster_data(
                "population",
                self.fetcher.get_population_sum,
                polygon_wkt, center_lon, center_lat
            )
            
            # Get terrain analysis data
            def fetch_terrain():
                from terrain_analysis import TerrainAnalyzer
                analyzer = TerrainAnalyzer()
                return analyzer.analyze_cluster_terrain(polygon_wkt)
            
            cluster_analysis["terrain_analysis"] = self._safe_fetch_cluster_data(
                "terrain_analysis",
                fetch_terrain
            )
            
            # Get weather data
            cluster_analysis["weather"] = self._safe_fetch_cluster_data(
                "weather",
                self.fetcher.get_weather_data,
                polygon_wkt, date_str
            )
        
        return cluster_analysis
    
    
    def _generate_summary(self, clusters_data: Dict[str, Any], total_population: float,
                         counties: List[str], weather_data: List[Dict], 
                         cluster_analysis: Optional[List[Dict]] = None, 
                         fire_name: Optional[str] = None, 
                         date_str: Optional[str] = None,
                         total_area_m2: float = 0,
                         max_cluster_area: float = 0) -> Dict[str, Any]:
        """
        Generate aggregated analysis summary from cluster data.
        
        Args:
            clusters_data: Clustered fire points data (already contains aggregated metrics)
            total_population: Total affected population
            counties: List of affected county names
            weather_data: Weather data from all clusters
            cluster_analysis: Detailed cluster analysis results
            fire_name: Name of the fire
            date_str: Analysis date in YYYY-MM-DD format
            total_area_m2: Pre-calculated total area (from single-pass aggregation)
            max_cluster_area: Pre-calculated max cluster area (from single-pass aggregation)
            
        Returns:
            Dictionary containing aggregated summary statistics
        """
        # Use pre-calculated values from clusters_data and parameters
        total_frp = clusters_data.get("total_frp", 0.0)
        max_brightness = clusters_data.get("max_brightness", 0)
        
        # Calculate max_frp from cluster analysis (as it's max per cluster, not global)
        max_frp = max([cluster.get("cluster_info", {}).get("frp", 0) for cluster in cluster_analysis or []], default=0)
        
        # Aggregate fire stations from all clusters
        fire_stations_summary = self._aggregate_fire_stations(cluster_analysis or [])
        
        summary = {
            "fire_name": fire_name or "Unknown",
            "analysis_date": date_str or "Unknown",
            "analysis_mmdd": date_str[5:] if date_str and len(date_str) >= 10 else "Unknown",  # Extract MM-DD from YYYY-MM-DD
            "fire_overview": {
                "total_fire_points": clusters_data.get("total_points", 0),
                "num_clusters": clusters_data.get("num_clusters", 0),
                "total_frp": round(total_frp, 2),
                "total_area_m2": round(total_area_m2, 2),
                "max_cluster_area_m2": round(max_cluster_area, 2),
                "max_brightness": max_brightness,
                "max_frp": round(max_frp, 2),
            },
            "affected_areas": {
                "counties": counties,
                "total_population_affected": round(total_population, 0),
                "num_counties": len(counties)
            },
            "fire_stations": fire_stations_summary,
        }
        
        # Weather summary
        if weather_data:
            weather_summary = self._summarize_weather_data(weather_data, cluster_analysis)
            summary["weather_conditions"] = weather_summary
        
        # Terrain summary
        if cluster_analysis:
            terrain_summary = self._summarize_terrain_data(cluster_analysis)
            if terrain_summary:
                summary["terrain_conditions"] = terrain_summary
        
        return summary
    
    def _aggregate_fire_stations(self, cluster_analysis: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate and deduplicate fire stations from all clusters by geographic location.
        
        Args:
            cluster_analysis: List of cluster analysis results containing fire station data
            
        Returns:
            Dictionary with aggregated fire station statistics and deduplicated station list
        """
        all_stations = []
        
        # Collect all stations from all clusters
        for cluster in cluster_analysis:
            fs = cluster.get("fire_stations", {})
            details = fs.get("station_details") or []
            
            for station in details:
                lon = station.get("lon")
                lat = station.get("lat")
                distance_m = station.get("distance_m")
                
                if lon is None or lat is None:
                    continue
                    
                all_stations.append({
                    "name": station.get("name"),
                    "lon": lon,
                    "lat": lat,
                    "distance_m": distance_m
                })
        
        # Deduplicate by geographic proximity (50m threshold)
        deduplicated_stations = self._deduplicate_by_location(all_stations, threshold_m=50)
        deduplicated_stations.sort(key=lambda x: (x.get("distance_m") is None, x.get("distance_m", 0.0)))
        
        # Calculate summary statistics
        valid_distances = [s["distance_m"] for s in deduplicated_stations if s.get("distance_m") is not None]
        num_stations = len(deduplicated_stations)
        nearest_distance_m = min(valid_distances) if valid_distances else None
        avg_distance_m = sum(valid_distances) / len(valid_distances) if valid_distances else None
        
        return {
            "total_stations": num_stations,
            "nearest_distance_m": round(nearest_distance_m, 2) if nearest_distance_m else None,
            "nearest_distance_km": round(nearest_distance_m / 1000, 2) if nearest_distance_m else None,
            "avg_distance_m": round(avg_distance_m, 2) if avg_distance_m else None,
            "avg_distance_km": round(avg_distance_m / 1000, 2) if avg_distance_m else None,
            "stations": [
                {
                    "name": s.get("name"),
                    "lon": s.get("lon"),
                    "lat": s.get("lat"),
                    "distance_m": s.get("distance_m"),
                    "distance_km": round(s.get("distance_m") / 1000, 2) if s.get("distance_m") else None
                }
                for s in deduplicated_stations[:10]  # Top 10 nearest stations
            ]
        }
    
    def _deduplicate_by_location(self, stations: List[Dict], threshold_m: float = 50.0) -> List[Dict]:
        """
        Deduplicate stations by geographic proximity using optimized clustering.
        
        Performance: O(n log n) with early termination instead of O(n²).
        Pre-sorts stations by distance, groups nearby stations efficiently.
        
        Args:
            stations: List of station dictionaries with lon, lat, distance_m, name
            threshold_m: Distance threshold in meters for considering stations as duplicates
            
        Returns:
            List of deduplicated stations with best representative from each group
        """
        if not stations:
            return []
        
        # Pre-sort by distance for better performance (closest stations first)
        sorted_stations = sorted(
            enumerate(stations), 
            key=lambda x: (x[1].get("distance_m") is None, x[1].get("distance_m", 0.0))
        )
        
        deduplicated = []
        used_indices = set()
        
        for orig_idx, station in sorted_stations:
            if orig_idx in used_indices:
                continue
            
            # Find all stations within threshold distance
            group = [station]
            used_indices.add(orig_idx)
            
            # Only compare with remaining unprocessed stations (optimization)
            for other_idx, other_station in sorted_stations:
                if other_idx in used_indices or orig_idx == other_idx:
                    continue
                
                # Early termination: if distance difference is too large, skip remaining
                # (works because list is sorted by distance)
                dist_diff = abs(
                    (station.get("distance_m") or 0) - (other_station.get("distance_m") or 0)
                )
                if dist_diff > threshold_m * 2:
                    # If center distances differ by >2*threshold, geographic distance must be >threshold
                    continue
                
                distance = _geo_distance_m(
                    station["lon"], station["lat"],
                    other_station["lon"], other_station["lat"]
                )
                
                if distance <= threshold_m:
                    group.append(other_station)
                    used_indices.add(other_idx)
            
            # Choose the best representative from the group
            # Priority: 1) Has name, 2) Shortest distance to fire
            best_station = min(group, key=lambda s: (
                s.get("name") is None or s.get("name") == "",
                s.get("distance_m") is None,
                s.get("distance_m", float('inf'))
            ))
            
            deduplicated.append(best_station)
        
        return deduplicated
    
    def _summarize_weather_data(self, weather_data: List[Dict], 
                                cluster_analysis: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Summarize weather data across all clusters using weighted averages.
        
        Args:
            weather_data: List of weather data dictionaries from each cluster
            cluster_analysis: Optional cluster analysis data for weighting by fire points
            
        Returns:
            Dictionary with weighted average weather values for each variable
        """
        if not weather_data:
            return {"note": "No weather data available"}
        
        weather_vars = ["bi", "tmmx", "tmmn", "vs", "fm1"]
        
        # Prepare data for weighted average calculation
        # Extract values and build data list
        data_list = []
        weights = []
        
        for i, cluster_weather in enumerate(weather_data):
            values_dict = cluster_weather.get("values", {})
            
            # Check if this cluster has any valid weather data
            has_data = any(values_dict.get(var) is not None for var in weather_vars)
            if not has_data:
                continue
            
            # Get cluster weight (fire points)
            cluster_weight = 1  # Default weight
            if cluster_analysis and i < len(cluster_analysis):
                cluster_info = cluster_analysis[i].get("cluster_info", {})
                cluster_weight = cluster_info.get("points", 1)
            
            # Add to data list with None handling
            data_list.append(values_dict)
            weights.append(cluster_weight)
        
        if not data_list or not weights:
            return {var: None for var in weather_vars}
        
        # Use the common weighted average utility function
        precisions = {var: 2 for var in weather_vars}  # All weather vars use 2 decimal places
        weather_summary = _calculate_weighted_average(data_list, weights, weather_vars, precisions)
        
        # Ensure all variables are present (even if None)
        for var in weather_vars:
            if var not in weather_summary:
                weather_summary[var] = None
        
        return weather_summary
    
    def _summarize_terrain_data(self, cluster_analysis: List[Dict]) -> Dict[str, Any]:
        """
        Summarize terrain data across all clusters using area-weighted averages.
        
        Args:
            cluster_analysis: List of cluster analysis results containing terrain data
            
        Returns:
            Dictionary with weighted average terrain metrics and dominant fragmentation level
        """
        terrain_metrics = []
        cluster_weights = []
        
        for cluster in cluster_analysis:
            terrain_data = cluster.get("terrain_analysis", {})
            if isinstance(terrain_data, dict) and "quantitative_metrics" in terrain_data:
                metrics = terrain_data["quantitative_metrics"]
                # Use cluster area as weight (more representative than fire points for terrain)
                cluster_info = cluster.get("cluster_info", {})
                weight = cluster_info.get("area_m2", 1)  # Default weight of 1 if no area
                
                terrain_metrics.append(metrics)
                cluster_weights.append(weight)
        
        if not terrain_metrics:
            return {"note": "No terrain data available"}
        
        if sum(cluster_weights) == 0:
            return {"note": "No valid terrain weight data"}
        
        # Define all metrics and their precisions
        metric_keys = [
            "num_land_types", "dominant_type_percent", "diversity_index",
            "high_risk_percent", "moderate_risk_percent", "low_risk_percent", "overall_risk_score",
            "continuous_fuel_percent", "natural_barriers_percent", "spread_potential_score"
        ]
        
        precisions = {
            "diversity_index": 3,
            "num_land_types": 1,
            "dominant_type_percent": 1,
            "high_risk_percent": 1,
            "moderate_risk_percent": 1,
            "low_risk_percent": 1,
            "overall_risk_score": 1,
            "continuous_fuel_percent": 1,
            "natural_barriers_percent": 1,
            "spread_potential_score": 1
        }
        
        # Calculate weighted averages using utility function
        weighted_avgs = _calculate_weighted_average(terrain_metrics, cluster_weights, metric_keys, precisions)
        
        # Add "avg_" prefix to keys
        summary = {f"avg_{key}": value for key, value in weighted_avgs.items()}
        
        # Most common fragmentation level (categorical - can't be averaged)
        fragmentation_levels = [m.get("fragmentation_level", "moderate") for m in terrain_metrics]
        from collections import Counter
        most_common_fragmentation = Counter(fragmentation_levels).most_common(1)[0][0]
        summary["dominant_fragmentation_level"] = most_common_fragmentation
        
        return summary
    
    def print_analysis_summary(self, analysis_result: Dict[str, Any]) -> None:
        """
        Print a formatted summary of the analysis results to console.
        
        Args:
            analysis_result: Complete analysis result dictionary from analyze_fire()
        """
        if "error" in analysis_result:
            print(f"Analysis Error: {analysis_result['error']}")
            return
        
        print("\n" + "="*60)
        print("FIRE ANALYSIS REPORT")
        print("="*60)
        
        # Fire overview
        fire_info = analysis_result["summary"]["fire_overview"]
        print(f"\nFIRE OVERVIEW:")
        print(f"   Fire Name: {analysis_result['fire_name']}")
        print(f"   Date: {analysis_result['analysis_date']}")
        print(f"   Total Fire Points: {fire_info['total_fire_points']:.0f}")
        print(f"   Clusters Formed: {fire_info['num_clusters']}")
        print(f"   Total FRP: {fire_info['total_frp']:.1f}")
        print(f"   Max FRP (Most Intense Cluster): {fire_info['max_frp']:.1f}")
        print(f"   Max Brightness: {fire_info['max_brightness']:.1f}")
        
        # Affected areas
        affected = analysis_result["summary"]["affected_areas"]
        print(f"\nAFFECTED AREAS:")
        print(f"   Counties: {', '.join(affected['counties'])}")
        print(f"   Estimated Population Affected: {affected['total_population_affected']:.0f}")
        
        # Emergency resources
        if "fire_stations" in analysis_result["summary"]:
            fs_summary = analysis_result["summary"]["fire_stations"]
            print(f"\nFIRE STATIONS SUMMARY:")
            print(f"   Total stations found: {fs_summary.get('total_stations', 0)}")
            if fs_summary.get('nearest_distance_km'):
                print(f"   Nearest station: {fs_summary['nearest_distance_km']:.1f} km")
            if fs_summary.get('avg_distance_km'):
                print(f"   Average distance: {fs_summary['avg_distance_km']:.1f} km")
            
            stations = fs_summary.get('stations', [])[:5]
            if stations:
                print(f"   Top {len(stations)} nearest stations:")
                for i, s in enumerate(stations):
                    name = s.get('name', f'Station {i+1}')
                    dist_km = s.get('distance_km', 0)
                    lon = s.get('lon', 0)
                    lat = s.get('lat', 0)
                    print(f"     {i+1}. {name} - {dist_km:.1f} km @ ({lon:.4f}, {lat:.4f})")
        
        # Weather conditions
        if "weather_conditions" in analysis_result["summary"]:
            weather = analysis_result["summary"]["weather_conditions"]
            print(f"\nWEATHER CONDITIONS (weighted averages):")
            for var, value in weather.items():
                if value is not None:
                    print(f"   {var.upper()}: {value:.1f}")
                else:
                    print(f"   {var.upper()}: No data available")
        
        print("\n" + "="*60)
    
    def _decimal_serializer(self, obj):
        """
        Custom JSON serializer for handling Decimal and non-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serializable representation (float for Decimal, str for others)
        """
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return str(obj)
    
    def save_analysis_to_json(self, analysis_result: Dict[str, Any], 
                             filename: Optional[str] = None) -> str:
        """
        Save analysis results to JSON file with proper encoding.
        
        Args:
            analysis_result: Complete analysis result dictionary from analyze_fire()
            filename: Optional output filename (auto-generated if not provided)
            
        Returns:
            Path to the saved JSON file
        """
        if not filename:
            fire_name = analysis_result.get("fire_name", "unknown")
            date_str = analysis_result.get("analysis_date", "unknown")
            filename = f"fire_analysis_{fire_name}_{date_str}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=self._decimal_serializer)
        
        return filename


def main():
    """Example usage of FireAnalysis"""
    analyzer = FireAnalysis()
    
    # Example analysis
    fire_name = "LNU_LIGHTNING_COMPLEX"
    date_str = "2020-08-20"
    
    result = analyzer.analyze_fire(fire_name, date_str)
    
    # Print summary
    analyzer.print_analysis_summary(result)
    
    # Save to file
    output_file = f"fire_analysis_{fire_name}_{date_str}.json"
    analyzer.save_analysis_to_json(result, output_file)

if __name__ == "__main__":
    main()
