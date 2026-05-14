from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

RAW_DATA_FILE = "day.csv"
SUBSAMPLED_FILE = "day_subsampled.csv"

COLUMN_ORDER = [
    "dteday", "cnt", "temp", "hum", "season", "yr", "mnth",
    "holiday", "weekday", "workingday", "weathersit", "atemp",
    "windspeed", "casual", "registered", "instant",
]

TARGET = "cnt"
FEATURES = ["temp", "hum"]

SUBSAMPLE_N = 360
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("bayes_bike")
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
