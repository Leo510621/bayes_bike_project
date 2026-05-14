"""
config.py — Global Configuration for Module B

Inherits data contracts from Module A (COLUMN_ORDER / TARGET / FEATURES / RANDOM_SEED),
extends module-specific constants for B (trace path, MCMC sampler hyperparameters).

Other modules import configurations via `from src.config import ...` to avoid hardcoding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

# Ensure output directories exist on startup
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Shared data constants with Module A
SUBSAMPLED_FILE = "day_subsampled.csv"
TARGET = "cnt"
FEATURES = ["temp", "hum"]
RANDOM_SEED = 42

# Output filenames for Module B
TRACE_FILE = "trace.nc"                       # For reading by Modules C/D
POSTERIOR_TABLE_FILE = "posterior_table.csv"  # Aligned with fields in A's frequentist_results.csv

# MCMC Sampler Hyperparameters
N_CHAINS = 4
N_TUNE = 1000          # warm-up (auto-tunes step size + acts as burn-in, discarded)
N_DRAWS = 2000         # 2000 samples retained per chain → 8000 total posterior samples
TARGET_ACCEPT = 0.95   # NUTS target acceptance rate; 0.95 is more stable than the default 0.8

np.random.seed(RANDOM_SEED)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project's unified logger to avoid duplicate handlers."""
    logger = logging.getLogger("bayes_bike_B")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
