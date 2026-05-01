"""
config.py — 项目全局配置

定义所有路径常量、数据契约（列顺序、目标变量、特征）、
超参数（子采样数量、随机种子），以及统一的日志设置。
其他模块通过 `from src.config import ...` 获取配置。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# --- 路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

# --- 数据契约 ---
RAW_DATA_FILE = "day.csv"
SUBSAMPLED_FILE = "day_subsampled.csv"

COLUMN_ORDER = [
    "dteday", "cnt", "temp", "hum", "season", "yr", "mnth",
    "holiday", "weekday", "workingday", "weathersit", "atemp",
    "windspeed", "casual", "registered", "instant",
]

TARGET = "cnt"
FEATURES = ["temp", "hum"]

# --- 超参数 ---
SUBSAMPLE_N = 360
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# --- 日志 ---
def setup_logging(level: str = "INFO") -> logging.Logger:
    """配置并返回项目统一 logger。"""
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
