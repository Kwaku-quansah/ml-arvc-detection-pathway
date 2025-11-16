"""
Global configuration for the ARVC ML project.

This keeps hard-coded paths and constants in one place.
"""

from pathlib import Path

# Random seed used across models
RANDOM_STATE: int = 42

# Project root = two levels up from this file (src/arvc_ml/config.py -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where figures and tables will be written
REPORTS_DIR = PROJECT_ROOT / "reports"
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TARGET_COL = "ARVC diagnosed"

