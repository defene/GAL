from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from database import (
    execute_single_query, execute_query, get_psycopg2_connection, get_db_cursor,
    wgs84_to_albers, get_point_in_polygon, 
    get_nearest_features, get_raster_value_at_point,
    get_polygon_population_sum, get_polygon_raster_stats, get_polygon_weather_data
)
from config import (
    ALBERS_SRID, TABLE_NAMES, WEATHER_TABLES, 
    SRID_CONFIG, DEFAULT_PARAMS, FIRE_CLUSTERING, FIRE_NAMES
)


class DBDataFetcher:
    """
    Universal database data fetcher for spatial queries.
    
    Provides high-level interface for querying fire-related spatial data including
    county information, fire stations, weather data, population, land cover, and
    fire point clustering.
    """
    
    def __init__(self) -> None:
        pass
    
    def get_county_name(self, lon: float, lat: float) -> Dict[str, Any]:
        """
        Get county information based on coordinates.
        
        Args:
            lon: Longitude in WGS84
            lat: Latitude in WGS84
            
        Returns:
            Dictionary with county_name, state, and optional note or error
        """
        try:
            result = get_point_in_polygon(
                table=TABLE_NAMES['county_boundary'],
                geom_column='wkb_geometry', 
                select_columns=['name'],
                lon=lon, lat=lat,
                src_srid=SRID_CONFIG['wgs84'],
                dst_srid=SRID_CONFIG['nad83']
            )
            
            county_name = result[0] if result else "Unknown"
            note = None if result else "Point not found in any county boundary"
            
            return {
                "county_name": county_name,
                "state": DEFAULT_PARAMS["default_state"],
                **({"note": note} if note else {})
            }
                
        except Exception as e:
            return {
                "county_name": "Unknown",
                "state": DEFAULT_PARAMS["default_state"],
                "error": str(e)
            }
    
    def get_nearest_fire_stations(self, lon: float, lat: float, limit: int = None) -> Dict[str, Any]:
        """
        Get nearest fire stations with coordinates and distances.
        
        Args:
            lon: Longitude in WGS84
            lat: Latitude in WGS84
            limit: Maximum number of stations to return (defaults to config value)
            
        Returns:
            Dictionary with stations list and station_details list
        """
        if limit is None:
            limit = DEFAULT_PARAMS["fire_station_limit"]
            
        try:
            results = get_nearest_features(
                table=TABLE_NAMES['fire_stations'],
                geom_column='wkb_geometry',
                select_columns=['name', 'ST_X(wkb_geometry)', 'ST_Y(wkb_geometry)'],
                lon=lon, lat=lat, limit=limit,
                src_srid=SRID_CONFIG['wgs84']
            )

            # Backward-compatible distances list
            distances: List[Optional[float]] = []
            station_details: List[Dict[str, Any]] = []
            for name, x_lon, y_lat, distance_m in results:
                dist_val = round(float(distance_m), 2) if distance_m is not None else None
                distances.append(dist_val)
                station_details.append({
                    "name": name or f"Station {len(station_details)+1}",
                    "lon": float(x_lon) if x_lon is not None else None,
                    "lat": float(y_lat) if y_lat is not None else None,
                    "distance_m": dist_val,
                })

            return {
                "stations": distances,              # kept for backward compatibility
                "station_details": station_details   # new detailed list with coordinates
            }
            
        except Exception as e:
            return {
                "stations": [],
                "error": str(e)
            }
    
    def get_weather_data(self, polygon_wkt: str, date: str) -> Dict[str, Any]:
        """
        Get weather data aggregated over polygon area for specific date.
        
        Args:
            polygon_wkt: Polygon in WKT format
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with date, values, missing variables, and optional note or error
        """
        try:
            dt = datetime.strptime(date, "%Y-%m-%d").date()
            
            weather_data = get_polygon_weather_data(
                polygon_wkt=polygon_wkt,
                date_value=dt.strftime("%Y-%m-%d"),
                weather_tables=WEATHER_TABLES,
                src_srid=SRID_CONFIG['wgs84']
            )
            
            return {
                "date": dt.isoformat(),
                "values": weather_data["values"],
                "missing": weather_data["missing"],
                "note": "Weather data aggregated over polygon area"
            }
            
        except Exception as e:
            return {
                "date": date,
                "values": {},
                "missing": list(WEATHER_TABLES.keys()),
                "error": str(e)
            }
    
    def get_population_sum(self, polygon_wkt: str, center_lon: float = None, center_lat: float = None) -> Dict[str, Any]:
        """
        Get population sum within polygon area with fallback support.
        
        Args:
            polygon_wkt: Polygon in WKT format
            center_lon: Center longitude for fallback buffer queries
            center_lat: Center latitude for fallback buffer queries
            
        Returns:
            Dictionary with pop_sum, method, and optional buffer_radius_m or error
        """
        try:
            # If center coordinates are provided, use the enhanced fallback method
            if center_lon is not None and center_lat is not None:
                from database import get_polygon_population_sum_with_fallback
                result = get_polygon_population_sum_with_fallback(
                    polygon_wkt, center_lon, center_lat, SRID_CONFIG['wgs84']
                )
                return result
            
            # Original method for backward compatibility
            pop_sum = get_polygon_population_sum(
                polygon_wkt=polygon_wkt,
                src_srid=SRID_CONFIG['wgs84']
            )
            
            if pop_sum is None or pop_sum <= 0:
                return {"pop_sum": 0.0, "note": "No valid population data in this polygon area"}
            else:
                return {"pop_sum": round(float(pop_sum), 2)}
                
        except Exception as e:
            return {"pop_sum": 0.0, "error": str(e)}
    
    def get_nlcd_stats(self, polygon_wkt: str) -> Dict[str, Any]:
        """
        Get NLCD land cover type statistical analysis within polygon area.
        
        Args:
            polygon_wkt: Polygon in WKT format
            
        Returns:
            Dictionary with bbox_wkt, stats, values, and optional note or error
        """
        try:
            raster_stats = get_polygon_raster_stats(
                polygon_wkt=polygon_wkt,
                table=TABLE_NAMES['nlcd'],
                raster_column='rast',
                src_srid=SRID_CONFIG['wgs84'],
                dst_srid=SRID_CONFIG['nlcd_raster']
            )
            
            if not raster_stats:
                return {
                    "error": "No NLCD raster data found in this polygon area"
                }
            
            return {
                "bbox_wkt": raster_stats["bbox_wkt"],
                "stats": raster_stats["stats"],
                "values": raster_stats["values"],
                "note": "NLCD raster statistics calculated for polygon area"
            }
            
        except Exception as e:
            return {
                "error": str(e)
            }
    
    def get_fire_clusters(self, fire_name: str, date_str: str) -> Dict[str, Any]:
        """
        Get fire point clusters for specific fire and date.
        
        Uses ST_ClusterDBSCAN for spatial clustering with adaptive parameters based on
        point count. Includes automatic merging of overlapping clusters.
        
        Args:
            fire_name: Fire name (must be in FIRE_NAMES config)
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with fire_name, date, total_points, num_clusters, clusters list,
            clustering_params, and optional note or error
        """
        try:
            with get_db_cursor() as cur:
                # Get summary statistics
                summary_result = self._get_fire_summary_stats(cur, fire_name, date_str)
                if summary_result.get("error"):
                    return summary_result
                
                # If no fire points, return early
                if summary_result["total_points"] == 0:
                    return summary_result
                
                # Get detailed cluster information
                clusters = self._execute_cluster_query(cur, fire_name, date_str, summary_result)
                
                # Post-process: merge overlapping clusters
                merged_clusters = self._merge_overlapping_clusters(clusters)
                
                # Update result with merged cluster data
                summary_result.update({
                    "num_clusters": len(merged_clusters),
                    "clusters": merged_clusters,
                    "original_clusters": len(clusters) if len(merged_clusters) != len(clusters) else None
                })
                
                return summary_result
                
        except Exception as e:
            # Return consistent error structure
            return self._build_error_result(fire_name, date_str, str(e))
    
    def _get_fire_summary_stats(self, cur, fire_name: str, date_str: str) -> Dict[str, Any]:
        """
        Get summary statistics for fire points without clustering.
        
        Args:
            cur: Database cursor
            fire_name: Fire name
            date_str: Date string
            
        Returns:
            Dictionary with summary stats and base result structure
        """
        where_clause = "local_date = %s AND fire = %s"
        params = [date_str, fire_name]
        
        # Dynamic clustering parameters
        eps_m = FIRE_CLUSTERING["eps_meters"]
        srid_m = FIRE_CLUSTERING["projection_srid"]
        
        # Get overall statistics
        summary_sql = self._build_summary_sql(where_clause)
        cur.execute(summary_sql, params)
        summary_row = cur.fetchone()
        
        # Determine clustering parameters based on actual point count
        total_points = int(summary_row[0] or 0) if summary_row else 0
        minpts = (FIRE_CLUSTERING["sparse_data_minpts"] if total_points < FIRE_CLUSTERING["sparse_data_threshold"] 
                 else FIRE_CLUSTERING["default_minpts"])
        
        clustering_note = f"Used {eps_m:.0f}m radius, min {minpts} points"
        if total_points < FIRE_CLUSTERING["sparse_data_threshold"]:
            clustering_note += " (adaptive clustering for sparse data)"
        
        # Build base result structure
        base_result = self._build_base_result(
            fire_name, date_str, summary_row, eps_m, minpts, srid_m, clustering_note
        )
        
        if not summary_row or not summary_row[0]:
            base_result["note"] = "No fire points found"
        
        return base_result
    
    def _build_summary_sql(self, where_clause: str) -> str:
        """
        Build SQL for fire summary statistics.
        
        Args:
            where_clause: WHERE clause for filtering
            
        Returns:
            SQL query string
        """
        return f"""
            SELECT COUNT(*) as total_points,
                   SUM(frp) as total_frp,
                   MAX(brightness_val) as max_brightness,
                   MAX(delta_t_val) as max_delta_t
            FROM {TABLE_NAMES['fire_daily_points']}
            WHERE {where_clause}
        """
    
    def _build_cluster_detail_sql(self, where_clause: str, eps_m: float, 
                                  minpts: int, srid_m: int) -> str:
        """
        Build SQL for detailed cluster information with DBSCAN clustering.
        
        Args:
            where_clause: WHERE clause for filtering
            eps_m: Epsilon parameter in meters
            minpts: Minimum points parameter
            srid_m: SRID for metric calculations
            
        Returns:
            SQL query string
        """
        buffer_radius = FIRE_CLUSTERING["single_point_buffer_radius"]
        
        return f"""
            WITH clustered AS (
                 SELECT frp as frp_val,
                      brightness_val as brightness_val,
                      delta_t_val as delta_t_val,
                      latitude as lat_val,
                      longitude as lon_val,
                      geom,
                      ST_ClusterDBSCAN(ST_Transform(geom, {srid_m}), {eps_m}, {minpts}) OVER() as cluster_id
                 FROM {TABLE_NAMES['fire_daily_points']}
                 WHERE {where_clause}
            ),
            cluster_stats AS (
                SELECT cluster_id,
                       COUNT(*) as points_in_cluster,
                       SUM(frp_val) as cluster_frp,
                       MAX(brightness_val) as cluster_max_brightness,
                       MAX(delta_t_val) as cluster_max_delta_t,
                       AVG(lat_val) as cluster_center_lat,
                       AVG(lon_val) as cluster_center_lon,
                       ST_Collect(geom) as cluster_geoms,
                       ST_Collect(ST_Transform(geom, {srid_m})) as cluster_geoms_proj
                FROM clustered
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
            )
            SELECT cluster_id,
                   points_in_cluster,
                   cluster_frp,
                   cluster_max_brightness,
                   cluster_max_delta_t,
                   cluster_center_lat,
                   cluster_center_lon,
                   CASE 
                       WHEN points_in_cluster = 1 THEN 
                           ST_AsText(ST_Transform(ST_Buffer(cluster_geoms_proj, {buffer_radius}), 4326))
                       WHEN points_in_cluster = 2 THEN 
                           ST_AsText(ST_Transform(ST_Buffer(ST_ConvexHull(cluster_geoms_proj), {buffer_radius}), 4326))
                       ELSE 
                           ST_AsText(ST_Transform(ST_ConvexHull(cluster_geoms_proj), 4326))
                   END as cluster_polygon_wkt,
                   CASE 
                       WHEN points_in_cluster = 1 THEN 
                           ST_Area(ST_Buffer(cluster_geoms_proj, {buffer_radius}))
                       WHEN points_in_cluster = 2 THEN 
                           ST_Area(ST_Buffer(ST_ConvexHull(cluster_geoms_proj), {buffer_radius}))
                       ELSE 
                           ST_Area(ST_ConvexHull(cluster_geoms_proj))
                   END as cluster_area_m2
            FROM cluster_stats
            ORDER BY cluster_id
        """
    
    def _build_base_result(self, fire_name: str, date_str: str, summary_row: Tuple,
                          eps_m: float, minpts: int, srid_m: int, 
                          clustering_note: str) -> Dict[str, Any]:
        """
        Build base result structure with summary statistics.
        
        Args:
            fire_name: Fire name
            date_str: Date string
            summary_row: Database query result row
            eps_m: Epsilon parameter
            minpts: Minimum points parameter
            srid_m: SRID for clustering
            clustering_note: Note about clustering parameters
            
        Returns:
            Base result dictionary
        """
        total_points = int(summary_row[0] or 0) if summary_row else 0
        
        return {
            "fire_name": fire_name,
            "date": date_str,
            "total_points": total_points,
            "num_clusters": 0,
            "total_frp": float(summary_row[1] or 0.0) if summary_row else 0.0,
            "max_brightness": float(summary_row[2] or 0.0) if summary_row else 0.0,
            "max_delta_t": float(summary_row[3] or 0.0) if summary_row else 0.0,
            "clusters": [],
            "clustering_params": {
                "eps_meters": eps_m,
                "min_points": minpts,
                "srid": srid_m
            },
            "note": clustering_note
        }
    
    def _build_error_result(self, fire_name: str, date_str: str, error_msg: str) -> Dict[str, Any]:
        """
        Build consistent error result structure.
        
        Args:
            fire_name: Fire name
            date_str: Date string
            error_msg: Error message
            
        Returns:
            Error result dictionary
        """
        return {
            "fire_name": fire_name,
            "date": date_str,
            "total_points": 0,
            "num_clusters": 0,
            "total_frp": 0.0,
            "max_brightness": 0.0,
            "max_delta_t": 0.0,
            "clusters": [],
            "clustering_params": {
                "eps_meters": FIRE_CLUSTERING["eps_meters"],
                "min_points": FIRE_CLUSTERING["default_minpts"],
                "srid": FIRE_CLUSTERING["projection_srid"]
            },
            "note": "Error occurred during clustering analysis",
            "error": error_msg
        }
    
    def _execute_cluster_query(self, cur, fire_name: str, date_str: str, 
                               summary_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute cluster detail query and process results.
        
        Args:
            cur: Database cursor
            fire_name: Fire name
            date_str: Date string
            summary_result: Summary result with clustering parameters
            
        Returns:
            List of cluster dictionaries
        """
        where_clause = "local_date = %s AND fire = %s"
        params = [date_str, fire_name]
        
        clustering_params = summary_result["clustering_params"]
        cluster_sql = self._build_cluster_detail_sql(
            where_clause,
            clustering_params["eps_meters"],
            clustering_params["min_points"],
            clustering_params["srid"]
        )
        
        cur.execute(cluster_sql, params)
        cluster_details = cur.fetchall()
        
        clusters = []
        for detail in cluster_details:
            cluster_id, points, frp, max_bright, max_delta, lat, lon, polygon_wkt, area_m2 = detail
            clusters.append({
                "cluster_id": int(cluster_id),
                "points": int(points),
                "frp": float(frp or 0.0),
                "max_brightness": float(max_bright or 0.0),
                "max_delta_t": float(max_delta or 0.0),
                "center_lat": float(lat or 0.0),
                "center_lon": float(lon or 0.0),
                "polygon_wkt": polygon_wkt,
                "area_m2": round(float(area_m2 or 0.0), 3)
            })
        
        return clusters
    
    def _merge_overlapping_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge clusters with overlapping or intersecting polygons.
        
        Uses Shapely for geometric operations. Falls back to original clusters
        if Shapely is unavailable.
        
        Args:
            clusters: List of cluster dictionaries with polygon_wkt
            
        Returns:
            List of merged cluster dictionaries
        """
        if len(clusters) <= 1:
            return clusters
        
        try:
            from shapely.wkt import loads as wkt_loads
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
        except ImportError:
            print("Warning: Shapely not available, skipping cluster merging")
            return clusters
        
        # Parse polygons and create cluster objects
        cluster_objects = self._parse_cluster_polygons(clusters)
        
        # Find and merge overlapping clusters
        merged_clusters = self._find_and_merge_groups(cluster_objects)
        
        # Add unmerged clusters (those without valid polygons)
        for obj in cluster_objects:
            if not obj['merged']:
                merged_clusters.append(obj['cluster'])
        
        # Reassign cluster IDs
        for i, cluster in enumerate(merged_clusters):
            cluster['cluster_id'] = i
        
        return merged_clusters
    
    def _parse_cluster_polygons(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse polygon WKT strings into Shapely geometry objects.
        
        Args:
            clusters: List of cluster dictionaries
            
        Returns:
            List of cluster objects with parsed polygons
        """
        from shapely.wkt import loads as wkt_loads
        
        cluster_objects = []
        for cluster in clusters:
            polygon_wkt = cluster.get('polygon_wkt')
            polygon = None
            
            if polygon_wkt:
                try:
                    parsed_polygon = wkt_loads(polygon_wkt)
                    if parsed_polygon.is_valid:
                        polygon = parsed_polygon
                except Exception:
                    pass  # Keep polygon as None
            
            cluster_objects.append({
                'cluster': cluster,
                'polygon': polygon,
                'merged': False
            })
        
        return cluster_objects
    
    def _find_and_merge_groups(self, cluster_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find overlapping cluster groups and merge them.
        
        Args:
            cluster_objects: List of cluster objects with polygons
            
        Returns:
            List of merged cluster dictionaries
        """
        merged_clusters = []
        
        for i, obj_i in enumerate(cluster_objects):
            if obj_i['merged'] or obj_i['polygon'] is None:
                continue
            
            # Start a new merge group with current cluster
            merge_group = [obj_i]
            obj_i['merged'] = True
            
            # Find all clusters that overlap with any cluster in the merge group
            self._expand_merge_group(merge_group, cluster_objects)
            
            # Create merged cluster from the group
            if len(merge_group) == 1:
                merged_clusters.append(merge_group[0]['cluster'])
            else:
                merged_cluster = self._create_merged_cluster(merge_group)
                merged_clusters.append(merged_cluster)
        
        return merged_clusters
    
    def _expand_merge_group(self, merge_group: List[Dict[str, Any]], 
                           cluster_objects: List[Dict[str, Any]]) -> None:
        """
        Expand merge group by finding all overlapping clusters iteratively.
        
        Args:
            merge_group: Current merge group (modified in place)
            cluster_objects: All cluster objects
        """
        changed = True
        while changed:
            changed = False
            for j, obj_j in enumerate(cluster_objects):
                if obj_j['merged'] or obj_j['polygon'] is None:
                    continue
                
                # Check if obj_j overlaps with any cluster in merge_group
                for group_obj in merge_group:
                    if group_obj['polygon'] and obj_j['polygon']:
                        try:
                            # Check for intersection or containment
                            if (group_obj['polygon'].intersects(obj_j['polygon']) and 
                                not group_obj['polygon'].touches(obj_j['polygon'])):
                                merge_group.append(obj_j)
                                obj_j['merged'] = True
                                changed = True
                                break
                        except Exception:
                            continue
                
                if changed:
                    break
    
    def _create_merged_cluster(self, merge_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a single cluster by merging multiple overlapping clusters.
        
        Args:
            merge_group: List of cluster objects to merge
            
        Returns:
            Merged cluster dictionary
        """
        try:
            from shapely.ops import unary_union
            
            # Extract cluster data
            clusters = [obj['cluster'] for obj in merge_group]
            polygons = [obj['polygon'] for obj in merge_group if obj['polygon']]
            
            # Merge polygons
            if polygons:
                try:
                    merged_polygon = unary_union(polygons)
                    merged_polygon_wkt = merged_polygon.wkt
                    # Sum original areas as approximation
                    merged_area_m2 = sum(c['area_m2'] for c in clusters)
                except Exception:
                    # Fallback: use the largest polygon
                    largest_polygon = max(polygons, key=lambda p: p.area)
                    merged_polygon_wkt = largest_polygon.wkt
                    merged_area_m2 = sum(c['area_m2'] for c in clusters)
            else:
                merged_polygon_wkt = clusters[0]['polygon_wkt']
                merged_area_m2 = clusters[0]['area_m2']
            
            # Aggregate cluster properties
            total_points = sum(c['points'] for c in clusters)
            total_frp = sum(c['frp'] for c in clusters)
            max_brightness = max(c['max_brightness'] for c in clusters)
            max_delta_t = max(c['max_delta_t'] for c in clusters)
            
            # Calculate weighted center
            total_weight = sum(c['points'] for c in clusters)
            if total_weight > 0:
                center_lat = sum(c['center_lat'] * c['points'] for c in clusters) / total_weight
                center_lon = sum(c['center_lon'] * c['points'] for c in clusters) / total_weight
            else:
                center_lat = sum(c['center_lat'] for c in clusters) / len(clusters)
                center_lon = sum(c['center_lon'] for c in clusters) / len(clusters)
            
            return {
                "cluster_id": 0,  # Will be reassigned later
                "points": total_points,
                "frp": round(total_frp, 2),
                "max_brightness": max_brightness,
                "max_delta_t": max_delta_t,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "polygon_wkt": merged_polygon_wkt,
                "area_m2": round(merged_area_m2, 3),
                "merged_from": len(clusters)  # Track how many clusters were merged
            }
            
        except Exception as e:
            print(f"Warning: Error merging clusters: {e}")
            # Fallback: return the largest cluster
            return max([obj['cluster'] for obj in merge_group], key=lambda c: c['points'])