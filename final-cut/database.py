import psycopg2
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Tuple, Callable
from config import DB_CONFIG, SRID_CONFIG, ALBERS_SRID, ALBERS_PROJ4, TABLE_NAMES


import pyproj
def transform_coordinates(lon, lat, src_crs, dst_crs):
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(lon, lat)

# ========== Database Connection Functions ==========

def get_psycopg2_connection():
    """Get psycopg2 database connection."""
    return psycopg2.connect(**DB_CONFIG)


def _execute_db_query(sql: str, params: Optional[tuple] = None, fetch_method: str = 'all') -> Any:
    """
    Execute database query with unified connection management.
    
    Args:
        sql: SQL query string
        params: Query parameters tuple
        fetch_method: 'all' for fetchall(), 'one' for fetchone()
        
    Returns:
        Query results based on fetch_method
    """
    conn = get_psycopg2_connection()
    try:
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        
        if fetch_method == 'one':
            result = cur.fetchone()
        else:
            result = cur.fetchall()
        
        cur.close()
        return result
    finally:
        conn.close()


def execute_query(sql, params=None):
    """
    Execute query and return all results.
    
    Args:
        sql: SQL query string
        params: Optional query parameters
        
    Returns:
        List of result tuples
    """
    return _execute_db_query(sql, params, 'all')


def execute_single_query(sql, params=None):
    """
    Execute query and return single result.
    
    Args:
        sql: SQL query string
        params: Optional query parameters
        
    Returns:
        Single result tuple or None
    """
    return _execute_db_query(sql, params, 'one')


# ========== Coordinate Transformation Tools ==========

def _coordinate_transform_fallback(x: float, y: float, src_srid: int, dst_srid: int) -> Tuple[float, float]:
    """
    Fallback coordinate transformation using PostGIS when pyproj unavailable.
    
    Args:
        x: X coordinate or longitude
        y: Y coordinate or latitude
        src_srid: Source SRID
        dst_srid: Destination SRID
        
    Returns:
        Transformed (x, y) coordinates
    """
    result = execute_single_query(f"""
        SELECT ST_X(ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), %s), %s)),
               ST_Y(ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), %s), %s))
    """, (x, y, src_srid, dst_srid, x, y, src_srid, dst_srid))
    return result[0], result[1] if result else (0, 0)


def wgs84_to_albers(lon: float, lat: float) -> Tuple[float, float]:
    """
    Convert WGS84 lon/lat to NLCD's custom Albers projection.
    
    Uses ALBERS_PROJ4 definition with pyproj.
    
    Args:
        lon: Longitude in WGS84
        lat: Latitude in WGS84
        
    Returns:
        (x, y) coordinates in Albers projection
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    aea_custom = pyproj.CRS.from_proj4(ALBERS_PROJ4)
    return transform_coordinates(lon, lat, wgs84, aea_custom)


# ========== Spatial Query Builders ==========

def build_point_geometry_sql(lon: float, lat: float, srid: int = None) -> str:
    """
    Build ST_SetSRID(ST_MakePoint()) SQL expression.
    
    Args:
        lon: Longitude
        lat: Latitude
        srid: Spatial reference ID (defaults to WGS84)
        
    Returns:
        SQL expression string
    """
    if srid is None:
        srid = SRID_CONFIG['wgs84']
    return f"ST_SetSRID(ST_MakePoint({lon}, {lat}), {srid})"


def build_contains_query_sql(table: str, geom_column: str, lon: float, lat: float, 
                           src_srid: int = None, dst_srid: int = None) -> str:
    """
    Build ST_Contains spatial query SQL.
    
    Args:
        table: Table name
        geom_column: Geometry column name
        lon: Longitude
        lat: Latitude
        src_srid: Source SRID (defaults to WGS84)
        dst_srid: Destination SRID for transformation
        
    Returns:
        SQL WHERE clause for ST_Contains
    """
    point_geom = build_point_geometry_sql(lon, lat, src_srid)
    if dst_srid and dst_srid != (src_srid or SRID_CONFIG['wgs84']):
        point_geom = f"ST_Transform({point_geom}, {dst_srid})"
    
    return f"ST_Contains({geom_column}, {point_geom})"


def build_distance_query_sql(table: str, geom_column: str, lon: float, lat: float,
                           src_srid: int = None, use_geography: bool = True) -> str:
    """
    Build ST_Distance spatial query SQL.
    
    Args:
        table: Table name
        geom_column: Geometry column name
        lon: Longitude
        lat: Latitude
        src_srid: Source SRID (defaults to WGS84)
        use_geography: Whether to use geography type for accurate distance
        
    Returns:
        SQL expression for distance calculation
    """
    point_geom = build_point_geometry_sql(lon, lat, src_srid)
    
    if use_geography:
        return f"ST_Distance({geom_column}::geography, {point_geom}::geography)"
    else:
        return f"ST_Distance({geom_column}, {point_geom})"


# ========== Database Connection Management ==========

@contextmanager
def get_db_cursor():
    """
    Context manager for database cursor with automatic cleanup.
    
    Yields:
        Database cursor
    """
    conn = get_psycopg2_connection()
    try:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        conn.close()


# ========== Common Spatial Query Templates ==========

def get_point_in_polygon(table: str, geom_column: str, select_columns: List[str], 
                        lon: float, lat: float, src_srid: int = None, dst_srid: int = None) -> Optional[tuple]:
    """
    Generic point-in-polygon query.
    
    Args:
        table: Table name
        geom_column: Geometry column name
        select_columns: List of columns to select
        lon: Longitude
        lat: Latitude
        src_srid: Source SRID
        dst_srid: Destination SRID
        
    Returns:
        First matching result tuple or None
    """
    columns = ', '.join(select_columns)
    contains_clause = build_contains_query_sql(table, geom_column, lon, lat, src_srid, dst_srid)
    
    sql = f"""
    SELECT {columns}
    FROM {table}
    WHERE {contains_clause}
    LIMIT 1
    """
    
    return execute_single_query(sql)


def get_nearest_features(table: str, geom_column: str, select_columns: List[str],
                        lon: float, lat: float, limit: int = 3, src_srid: int = None) -> List[tuple]:
    """
    Generic nearest features query.
    
    Args:
        table: Table name
        geom_column: Geometry column name
        select_columns: List of columns to select
        lon: Longitude
        lat: Latitude
        limit: Maximum number of results
        src_srid: Source SRID
        
    Returns:
        List of result tuples with distances
    """
    columns = ', '.join(select_columns)
    distance_expr = build_distance_query_sql(table, geom_column, lon, lat, src_srid)
    point_geom = build_point_geometry_sql(lon, lat, src_srid)
    
    sql = f"""
    SELECT {columns}, {distance_expr} as distance_meters
    FROM {table}
    WHERE {geom_column} IS NOT NULL
    ORDER BY {geom_column}::geography <-> {point_geom}::geography
    LIMIT %s
    """
    
    return execute_query(sql, (limit,))


def get_raster_value_at_point(table: str, raster_column: str, lon: float, lat: float,
                             date_column: str = None, date_value: str = None) -> Optional[float]:
    """
    Generic raster value extraction at point.
    
    Args:
        table: Table name
        raster_column: Raster column name
        lon: Longitude
        lat: Latitude
        date_column: Optional date column name
        date_value: Optional date value for filtering
        
    Returns:
        Raster value at point or None
    """
    point_geom = build_point_geometry_sql(lon, lat, SRID_CONFIG['wgs84'])
    
    where_clause = ""
    params = [lon, lat]
    
    if date_column and date_value:
        where_clause = f"WHERE t.{date_column}::date = %s::date AND "
        params.append(date_value)
    else:
        where_clause = "WHERE "
    
    sql = f"""
    WITH pt AS (SELECT {point_geom} AS g),
         tile AS (SELECT t.{raster_column} FROM {table} t, pt
                 {where_clause}ST_Intersects(t.{raster_column}, ST_Transform(pt.g, ST_SRID(t.{raster_column})))
                 LIMIT 1)
    SELECT ST_Value(tile.{raster_column}, ST_Transform((SELECT g FROM pt), ST_SRID(tile.{raster_column})))::double precision
    FROM tile
    """
    
    result = execute_single_query(sql, tuple(params))
    return result[0] if result else None


# ========== Polygon Spatial Query Tools ==========

def get_polygon_population_sum(polygon_wkt: str, src_srid: int = None) -> Optional[float]:
    """
    Calculate population sum within polygon area.
    
    Args:
        polygon_wkt: Polygon in WKT format
        src_srid: Source SRID (defaults to WGS84)
        
    Returns:
        Population sum or 0.0
    """
    if src_srid is None:
        src_srid = SRID_CONFIG['wgs84']
    
    # Transform polygon to population raster SRID for accurate calculation
    dst_srid = SRID_CONFIG['ca_albers']  # Population raster uses CA Albers
    
    sql = f"""
    WITH poly AS (
        SELECT ST_Transform(ST_GeomFromText(%s, %s), %s) AS geom
    )
    SELECT SUM((stats).sum) AS pop_sum
    FROM (
        SELECT ST_SummaryStats(ST_Clip(rast, poly.geom), 1, TRUE) AS stats
        FROM {TABLE_NAMES['population']}, poly
        WHERE ST_Intersects(rast, poly.geom)
    ) t
    """
    
    result = execute_single_query(sql, (polygon_wkt, src_srid, dst_srid))
    return float(result[0]) if result and result[0] else 0.0


def _convert_polygon_wkt_to_albers(polygon_wkt: str) -> Optional[str]:
    """
    Convert WGS84 polygon WKT to Albers projection.
    
    Args:
        polygon_wkt: Polygon in WKT format (WGS84)
        
    Returns:
        Polygon in Albers WKT format or None on error
    """
    try:
        import re
        
        # Extract coordinates from WKT
        coords_match = re.search(r'POLYGON\(\(([^)]+)\)\)', polygon_wkt)
        if not coords_match:
            return None
        
        coords_str = coords_match.group(1)
        coord_pairs = coords_str.split(',')
        
        # Convert each coordinate pair
        albers_coords = []
        for coord_pair in coord_pairs:
            lon, lat = map(float, coord_pair.strip().split())
            albers_x, albers_y = wgs84_to_albers(lon, lat)
            albers_coords.append(f"{albers_x} {albers_y}")
        
        # Create Albers WKT
        return f"POLYGON(({','.join(albers_coords)}))"
        
    except Exception as e:
        print(f"Error converting polygon coordinates: {e}")
        return None


def _process_nlcd_polygon_raster(polygon_wkt: str, table: str, raster_column: str) -> Optional[dict]:
    """
    Process NLCD polygon raster query with custom Albers projection.
    
    Args:
        polygon_wkt: Polygon in WKT format (WGS84)
        table: NLCD table name
        raster_column: Raster column name
        
    Returns:
        Dictionary with bbox_wkt, stats, and values or None
    """
    # Convert WGS84 polygon to Albers projection
    albers_wkt = _convert_polygon_wkt_to_albers(polygon_wkt)
    if not albers_wkt:
        return None
    
    # For NLCD data with SRID=0, we need to handle the geometry creation differently
    sql = f"""
    WITH poly AS (
        SELECT ST_SetSRID(ST_GeomFromText(%s), {ALBERS_SRID}) AS geom
    ),
    clipped_rasters AS (
        SELECT ST_Clip(t.{raster_column}, poly.geom) AS clipped_rast
        FROM {table} t, poly
        WHERE ST_Intersects(t.{raster_column}, poly.geom)
    ),
    merged_raster AS (
        SELECT ST_Union(clipped_rast) AS merged_rast
        FROM clipped_rasters
    )
    SELECT 
        ST_AsText(ST_Envelope(merged_rast)) AS bbox_wkt,
        ST_SummaryStats(merged_rast, 1, TRUE) AS stats,
        ST_DumpValues(merged_rast, 1, TRUE) AS values
    FROM merged_raster
    WHERE merged_rast IS NOT NULL
    """
    
    return execute_single_query(sql, (albers_wkt,))


def _process_standard_polygon_raster(polygon_wkt: str, table: str, raster_column: str,
                                     src_srid: int, dst_srid: int) -> Optional[tuple]:
    """
    Process standard polygon raster query with SRID transformation.
    
    Args:
        polygon_wkt: Polygon in WKT format
        table: Table name
        raster_column: Raster column name
        src_srid: Source SRID
        dst_srid: Destination SRID
        
    Returns:
        Result tuple or None
    """
    sql = f"""
    WITH poly AS (
        SELECT CASE 
            WHEN %s = 0 THEN ST_GeomFromText(%s, %s) 
            ELSE ST_Transform(ST_GeomFromText(%s, %s), %s) 
        END AS geom
    ),
    clipped_rasters AS (
        SELECT ST_Clip(t.{raster_column}, poly.geom) AS clipped_rast
        FROM {table} t, poly
        WHERE ST_Intersects(t.{raster_column}, poly.geom)
    ),
    merged_raster AS (
        SELECT ST_Union(clipped_rast) AS merged_rast
        FROM clipped_rasters
    )
    SELECT 
        ST_AsText(ST_Envelope(merged_rast)) AS bbox_wkt,
        ST_SummaryStats(merged_rast, 1, TRUE) AS stats,
        ST_DumpValues(merged_rast, 1, TRUE) AS values
    FROM merged_raster
    WHERE merged_rast IS NOT NULL
    """
    
    return execute_single_query(sql, (dst_srid, polygon_wkt, src_srid, polygon_wkt, src_srid, dst_srid))


def get_polygon_raster_stats(polygon_wkt: str, table: str, raster_column: str = 'rast', 
                           src_srid: int = None, dst_srid: int = None) -> Optional[dict]:
    """
    Get raster statistics within polygon area.
    
    Handles both NLCD (custom Albers, SRID=0) and standard raster data.
    
    Args:
        polygon_wkt: Polygon in WKT format
        table: Table name
        raster_column: Raster column name (default 'rast')
        src_srid: Source SRID (defaults to WGS84)
        dst_srid: Destination SRID (defaults to ALBERS_SRID)
        
    Returns:
        Dictionary with bbox_wkt, stats, and values or None
    """
    if src_srid is None:
        src_srid = SRID_CONFIG['wgs84']
    if dst_srid is None:
        dst_srid = ALBERS_SRID
    
    # For NLCD data (ca_clipped table), use coordinate transformation
    if table == TABLE_NAMES['nlcd']:
        result = _process_nlcd_polygon_raster(polygon_wkt, table, raster_column)
    else:
        result = _process_standard_polygon_raster(polygon_wkt, table, raster_column, src_srid, dst_srid)
    
    if not result:
        return None
    
    bbox_wkt, stats, values = result
    return {
        "bbox_wkt": bbox_wkt,
        "stats": stats,
        "values": values
    }


def get_polygon_weather_data(polygon_wkt: str, date_value: str, weather_tables: dict,
                           src_srid: int = None) -> dict:
    """
    Get weather data aggregated over polygon area.
    
    Args:
        polygon_wkt: Polygon in WKT format
        date_value: Date value for filtering
        weather_tables: Dictionary mapping variable names to table names
        src_srid: Source SRID (defaults to WGS84)
        
    Returns:
        Dictionary with 'values' and 'missing' keys
    """
    if src_srid is None:
        src_srid = SRID_CONFIG['wgs84']
    
    results = {}
    missing = []
    
    for var, table in weather_tables.items():
        try:
            sql = f"""
            WITH poly AS (
                SELECT ST_GeomFromText(%s, %s) AS geom
            ),
            clipped_data AS (
                SELECT ST_Clip(t.rast, ST_Transform(poly.geom, ST_SRID(t.rast))) AS clipped_rast
                FROM {table} t, poly
                WHERE t.dt::date = %s::date 
                  AND ST_Intersects(t.rast, ST_Transform(poly.geom, ST_SRID(t.rast)))
            ),
            merged_data AS (
                SELECT ST_Union(clipped_rast) AS merged_rast
                FROM clipped_data
            )
            SELECT 
                (ST_SummaryStats(merged_rast, 1, TRUE)).mean AS avg_value,
                (ST_SummaryStats(merged_rast, 1, TRUE)).min AS min_value,
                (ST_SummaryStats(merged_rast, 1, TRUE)).max AS max_value,
                (ST_SummaryStats(merged_rast, 1, TRUE)).count AS pixel_count
            FROM merged_data
            WHERE merged_rast IS NOT NULL
            """
            
            result = execute_single_query(sql, (polygon_wkt, src_srid, date_value))
            
            if result and result[0] is not None:
                avg_val, min_val, max_val, pixel_count = result
                results[var] = float(avg_val) if avg_val else None
            else:
                results[var] = None
                missing.append(var)
                
        except Exception:
            results[var] = None
            missing.append(var)
    
    return {"values": results, "missing": missing}


# ========== Single Point Processing Fallback Query ==========

def get_point_buffer_polygon(lon: float, lat: float, radius_m: float, 
                           src_srid: int = None, dst_srid: int = None) -> Optional[str]:
    """
    Create buffer polygon around point with proper coordinate transformation.
    
    Args:
        lon: Longitude
        lat: Latitude
        radius_m: Buffer radius in meters
        src_srid: Source SRID (defaults to WGS84)
        dst_srid: Destination SRID (defaults to WGS84)
        
    Returns:
        Polygon WKT string or None on error
    """
    if src_srid is None:
        src_srid = SRID_CONFIG['wgs84']
    if dst_srid is None:
        dst_srid = SRID_CONFIG['wgs84']
    
    try:
        # For accurate buffering, transform to projected coordinate system
        proj_srid = SRID_CONFIG['ca_albers']  # Use CA Albers for accurate meter-based buffering
        
        sql = f"""
        WITH point_proj AS (
            SELECT ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), %s), %s) AS geom
        ),
        buffered AS (
            SELECT ST_Buffer(geom, %s) AS geom FROM point_proj
        )
        SELECT ST_AsText(ST_Transform(geom, %s)) AS polygon_wkt
        FROM buffered
        """
        
        result = execute_single_query(sql, (lon, lat, src_srid, proj_srid, radius_m, dst_srid))
        return result[0] if result and result[0] else None
        
    except Exception as e:
        print(f"Error creating buffer polygon: {e}")
        return None


def get_polygon_population_sum_with_fallback(polygon_wkt: str, center_lon: float, center_lat: float, 
                                           src_srid: int = None) -> dict:
    """
    Get population sum with fallback to point buffer if polygon fails.
    
    Tries original polygon first, then falls back to point buffers with decreasing radii.
    
    Args:
        polygon_wkt: Polygon in WKT format
        center_lon: Center longitude for fallback
        center_lat: Center latitude for fallback
        src_srid: Source SRID (defaults to WGS84)
        
    Returns:
        Dictionary with pop_sum, method, and optional buffer_radius_m
    """
    if src_srid is None:
        src_srid = SRID_CONFIG['wgs84']
    
    # First try the original polygon
    try:
        pop_sum = get_polygon_population_sum(polygon_wkt, src_srid)
        if pop_sum and pop_sum > 0:
            return {"pop_sum": round(float(pop_sum), 2), "method": "polygon"}
    except Exception as e:
        print(f"Polygon query failed: {e}")
    
    # Fallback to point buffer with different radii
    from config import FIRE_CLUSTERING
    buffer_radii = [
        FIRE_CLUSTERING["single_point_buffer_radius"],
        1000,  # 1km fallback
        500    # 500m fallback
    ]
    
    for radius in buffer_radii:
        try:
            buffer_wkt = get_point_buffer_polygon(center_lon, center_lat, radius, src_srid, src_srid)
            if buffer_wkt:
                pop_sum = get_polygon_population_sum(buffer_wkt, src_srid)
                if pop_sum and pop_sum > 0:
                    return {
                        "pop_sum": round(float(pop_sum), 2),
                        "method": "point_buffer",
                        "buffer_radius_m": radius
                    }
        except Exception:
            continue
    
    return {"pop_sum": 0.0, "method": "failed", "note": "All query methods failed"}