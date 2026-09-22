from pathlib import Path
from typing import Dict, List

# Core Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
PREDICTION_DIR = DATA_DIR / "predictions"

# Ensure all directories exist dynamically on the mounted disk at runtime
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, PREDICTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Historical Training Range and Current Operational Year
START_YEAR = 2020
CURRENT_YEAR = 2026

# Model Determinism
RANDOM_STATE = 42

# 20 Targeted Player Prop Markets and their statistical distribution types
PROP_CONFIG: Dict[str, Dict[str, str]] = {
    # Passing (6 Props)
    "passing_yards": {"type": "continuous", "distribution": "tweedie"},
    "passing_attempts": {"type": "count", "distribution": "negative_binomial"},
    "passing_completions": {"type": "count", "distribution": "negative_binomial"},
    "passing_touchdowns": {"type": "count", "distribution": "poisson"},
    "passing_interceptions": {"type": "count", "distribution": "poisson"},
    "passing_longest": {"type": "continuous", "distribution": "extreme_value"},
    
    # Rushing (4 Props)
    "rushing_yards": {"type": "continuous", "distribution": "tweedie"},
    "rushing_attempts": {"type": "count", "distribution": "negative_binomial"},
    "rushing_touchdowns": {"type": "count", "distribution": "poisson"},
    "rushing_longest": {"type": "continuous", "distribution": "extreme_value"},
    
    # Receiving (5 Props)
    "receiving_yards": {"type": "continuous", "distribution": "tweedie"},
    "receptions": {"type": "count", "distribution": "negative_binomial"},
    "targets": {"type": "count", "distribution": "negative_binomial"},
    "receiving_touchdowns": {"type": "count", "distribution": "poisson"},
    "receiving_longest": {"type": "continuous", "distribution": "extreme_value"},
    
    # Specialists & Scoring (3 Props)
    "anytime_td": {"type": "binary", "distribution": "bernoulli"},
    "field_goals_made": {"type": "count", "distribution": "poisson"},
    "extra_points_made": {"type": "count", "distribution": "poisson"},
    
    # Defensive / IDP (2 Props)
    "defensive_sacks": {"type": "count", "distribution": "poisson"},
    "defensive_tackles_combined": {"type": "count", "distribution": "negative_binomial"}
}

PROP_TARGETS: List[str] = list(PROP_CONFIG.keys())
