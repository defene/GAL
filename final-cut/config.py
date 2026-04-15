# Database configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read database configuration from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "spatialdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")  # Default for development only

# NLCD color mapping
NLCD_COLORMAP = {
    11: (70, 107, 159), 12: (209, 222, 248),
    21: (222, 197, 197), 22: (255, 170, 127),
    23: (255, 112, 66), 24: (255, 0, 0),
    31: (179, 174, 163), 41: (104, 171, 95),
    42: (28, 95, 44), 43: (181, 197, 143),
    52: (204, 184, 121), 71: (223, 223, 194),
    81: (220, 217, 57), 82: (171, 108, 40),
    90: (184, 217, 235), 95: (108, 159, 184),
    250: (0, 0, 0)
}

# Projection parameters
ALBERS_PROJ4 = (
    "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
    "+datum=WGS84 +units=m +no_defs"
)
ALBERS_SRID = 0  # NLCD raster SRID is 0 (undefined coordinate system)

# Database connection parameters (using environment variables)
DB_CONFIG = {
    "dbname": DB_NAME,
    "user": DB_USER, 
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": int(DB_PORT)
}

# Database table name configuration
TABLE_NAMES = {
    "county_boundary": "ca_boundary_geom",
    "fire_stations": "fire_stations", 
    "fire_daily_points": "fire_daily_points",
    "population": "capop_2020_100m",
    "nlcd": "ca_clipped"
}

# Weather data table name mapping
WEATHER_TABLES = {
    "bi": "bi_2020_all_days_ca",
    "tmmx": "tmmx_2020_all_days_ca", 
    "tmmn": "tmmn_2020_all_days_ca",
    "vs": "vs_2020_all_days_ca",
    "fm1": "fm1_2020_all_days_ca"
}

# SRID configuration
SRID_CONFIG = {
    "wgs84": 4326,
    "nad83": 4269, 
    "ca_albers": 3310,
    "population_raster": 3310,
    "weather_raster": 4326,  # Weather raster data SRID
    "nlcd_raster": 0         # NLCD raster data SRID (undefined)
}

# Default query parameters
DEFAULT_PARAMS = {
    "fire_station_limit": 3,
    "default_state": "CA"
}

# Fire clustering parameters
FIRE_CLUSTERING = {
    "eps_meters": 2500.0,  # Reduced from 3000m to 800m for better clustering
    "default_minpts": 3,  # Reduced from 3 to 2 for more sensitive clustering
    "sparse_data_minpts": 1,
    "sparse_data_threshold": 10,
    "projection_srid": 3310,
    "single_point_buffer_radius": 2500.0  # Buffer radius for single point clusters (consistent with eps_meters)
}

# Unit system configuration - Support for A/B/C experiment groups
UNIT_SYSTEM_CONFIG = {
    "current_system": "B",  # Default to group B (standard units)
    "systems": {
        "A": {
            "name": "Small Values",
            "description": "km², °C, m/s, mile",
            "area": "km2",
            "distance": "mile",
            "temperature": "C", 
            "speed": "ms"
        },
        "B": {
            "name": "Standard",
            "description": "acres, ℉, m/s, mile",
            "area": "acres",
            "distance": "mile", 
            "temperature": "F",
            "speed": "ms"
        },
        "C": {
            "name": "Large Values", 
            "description": "m², K, km/h, km",
            "area": "m2",
            "distance": "km",
            "temperature": "K",
            "speed": "kmh"
        }
    }
}

# Known fire name list
FIRE_NAMES = [
    "AUGUST_COMPLEX",
    "CREEK", 
    "CZU_AUG_LIGHTNING",
    "DOLAN",
    "EL_DORADO",
    "FORK",
    "LNU_LIGHTNING_COMPLEX",
    "NORTH_COMPLEX",
    "RED_SALMON_COMPLEX",
    "SCU_LIGHTNING_COMPLEX",
    "SLATER",
    "SLINK",
    "SQF_COMPLEX",
    "WOODWARD"
]

# Prompt template configuration - Two independent templates
PROMPT_SECTION_CONFIG = {
    "current_template": "template2",  # Default template: template1 or template2
    
    "templates": {
        "template1": {
            "fire_analysis": {
                "system": {
                    "order": [
                        "role",
                        "simplified_global_guidelines",
                        "core_principles",
                        "rag_guidance",
                        "output_schema",
                        "format_rules"
                    ]
                },
                "user": {
                    "order": [
                        "previous_context?",
                        "cumulative_context?",
                        "fire_overview",
                        "affected_areas?",
                        "global_change_hints?",
                        "rag_context?",
                        "clusters",
                        "instruction_recall"
                    ],
                    "toggles": {
                        "terrain_format": "detailed",
                        "include_environmental_info": True,
                        "cluster_display_mode": "highlights"
                    }
                }
            }
        },
        
        "template2": {
            "fire_analysis": {
                "system": {
                    "order": [
                        "role",
                        "global_guidelines",
                        "resource_principles",
                        "analysis_approach",
                        "historical_context?",
                        "output_schema",
                        "format_rules"
                    ],
                    "toggles": {
                        "include_historical_guidelines": "auto"
                    }
                },
                "user": {
                    "order": [
                        "previous_context?",
                        "cumulative_context?",
                        "fire_overview",
                        "affected_areas?",
                        "global_change_hints?",
                        "rag_context?",
                        "clusters"
                    ],
                    "toggles": {
                        "terrain_format": "detailed",
                        "include_environmental_info": True,
                        "cluster_display_mode": "details"
                    }
                }
            }
        }
    }
}

# RAG system configuration
RAG_CONFIG = {
    "corpus_glob": "fire_analysis_*.json",
    "weights": "uniform",
    "gt_columns": ["EST_IM_COST_TO_DATE_FIXED_DAILY", "TOTAL_PERSONNEL", "EST_IM_COST_TO_DATE_FIXED"],
    "output_dir": "rag_data/",
    "date_cycle_days": 366,
    "unique_by_fire": True,  # Whether to deduplicate by fire name, ensuring top-k results from different fires
    "display_format": "detailed"  # Display format: "range" (concise ranges) | "detailed" (detailed entries)
}

# RAG data split configuration
RAG_SPLIT = {
    "rag_fires": ["CREEK", "Dolan", "FORK", "North_Complex", "Red_Salmon_Complex", "SLATER", "Slink", "SQF_COMPLEX", "WOODWARD", "LNU_LIGHTNING_COMPLEX"],
    "test_fires": ["SCU_LIGHTNING_COMPLEX", "CZU_AUG_LIGHTNING", "EL_DORADO", "AUGUST_COMPLEX"],
    "default_policy": "exclude",
    "strict_no_overlap": True
}

# Prompt plugin configuration
PROMPT_PLUGINS = {
    "rag": {
        "enabled": True,                    # Whether to enable RAG injection
        "provider": "dual",                 # Provider type: "dual" (dual RAG), "retriever" (original)
        "top_k": 5,                        # Number of similar samples to retrieve
        "fail_mode": "silent",             # Failure mode: silent=skip silently, error=raise exception
        "task_modes": ["fire_analysis"]   # Supported task modes
    }
}

# LLM model configuration
LLM_CONFIG = {
    "default_model": "gpt-4o-mini",
    "default_temperature": 0.0,
    "default_max_tokens": 15000,
    "default_max_retries": 3,
    "supported_models": {
        "gpt-4o-mini": {
            "token_param": "max_tokens",
            "supported_temperature": [0.0, 1.0, 2.0],
            "default_temperature": 0.0
        },
        "gpt-4o": {
            "token_param": "max_tokens",
            "supported_temperature": [0.0, 1.0, 2.0],
            "default_temperature": 0.0
        },
        "gpt-5-mini": {
            "token_param": "max_completion_tokens",
            "supported_temperature": [1.0],
            "default_temperature": 1.0
        },
        "gpt-5": {
            "token_param": "max_completion_tokens",
            "supported_temperature": [1.0],
            "default_temperature": 1.0
        },
        "o3-mini": {
            "token_param": "max_completion_tokens",
            "supported_temperature": [1.0],
            "default_temperature": 1.0
        },
        "o3": {
            "token_param": "max_completion_tokens",
            "supported_temperature": [1.0],
            "default_temperature": 1.0
        },
        "gpt-4.1-mini": {
            "token_param": "max_completion_tokens",
            "supported_temperature": [0.0],
            "default_temperature": 0.0
        },
        "gpt-4.1": {
            "token_param": "max_completion_tokens",
            "supported_temperature": [1.0],
            "default_temperature": 1.0
        },
        "gemini-2.5-flash": {
            "token_param": "max_output_tokens",
            "supported_temperature": [0.0, 1.0, 2.0],
            "default_temperature": 0.0
        },
        "gemini-2.5-pro": {
            "token_param": "max_output_tokens",
            "supported_temperature": [0.0, 1.0, 2.0],
            "default_temperature": 0.0
        }
    }
}

# Cumulative data configuration
CUMULATIVE_CONFIG = {
    "enabled": True,                         # Whether to enable cumulative context
    "rolling_windows": [3, 7],              # Rolling window days (configurable)
    "precision": {
        "cost": 0,                          # Cost rounded to dollars
        "personnel": 0,                     # Personnel rounded to persons
        "days": 0                           # Days rounded to integer
    },
    "display_format": "bullet_points"       # Display format: "bullet_points" | "json_fragment"
}

# Comparison configuration - Global switches and thresholds
COMPARISON_CONFIG = {
    "enabled": True,
    "baseline_policy": "yesterday",           # "yesterday" | "last_with_fire_points"
    "use_arrows": True,
    "show_percent_change": True,
    "percent_min_change": 0.05,              # Changes below 5% considered insignificant
    "float_precision": 1,
    "percent_precision": 1,
    "suppress_zero_zero_details": True,      # Only show summary when both today and yesterday are zero
    "no_fire_policy": "drop_to_zero_with_notice",  # See above
    "show_fire_stations_when_no_fire": False,
    "metrics": {
        "fire_overview": ["total_fire_points","num_clusters","total_frp","total_area_m2","max_frp","max_brightness"],
        "affected_areas": ["counties","total_population_affected","num_counties"],
        "fire_stations": ["total_stations","nearest_distance_m"]
    }
}

# Run configuration - Specify fires to process and output settings
RUN_FIRES = ["LNU_LIGHTNING_COMPLEX", "SCU_LIGHTNING_COMPLEX", "CZU_AUG_LIGHTNING", "EL_DORADO", "AUGUST_COMPLEX"]  # List of fires to process, corresponds to final_data_cleaned/<name>_gt.csv
# RUN_FIRES = ["AUGUST_COMPLEX"]

OUTPUT_ROOT = "runs"  # Run output root directory

# Rolling window configuration
ROLLING_CONFIG = {
    "windows": [3, 7],  # Rolling window sizes (days)
    "min_periods": 1,   # Minimum required data points (1 means calculate even with insufficient window)
    "trend_windows": [3, 7],  # Trend slope calculation windows
    "trend_min_points": 2,    # Minimum points for trend calculation
    "ratio_mode": "to_date_max"  # Ratio mode: current value / max since fire start
}
