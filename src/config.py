"""
config.py — Module B 全局配置

继承 A 的数据契约（COLUMN_ORDER / TARGET / FEATURES / RANDOM_SEED），
扩展 B 模块特有的常量（trace 路径、MCMC 采样器超参数）。

其他模块通过 `from src.config import ...` 获取配置，避免硬编码。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

# 启动时确保输出目录存在
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 与 A 模块共享数据
SUBSAMPLED_FILE = "day_subsampled.csv"
TARGET = "cnt"
FEATURES = ["temp", "hum"]
RANDOM_SEED = 42

# B 模块输出文件名 
TRACE_FILE = "trace.nc"                       # 供 C/D 模块读取
POSTERIOR_TABLE_FILE = "posterior_table.csv"  # 与 A 的 frequentist_results.csv 字段对齐

# MCMC 采样器超参数 
N_CHAINS = 4
N_TUNE = 1000          # warm-up（自动调步长 + 充当 burn-in，丢弃）
N_DRAWS = 2000         # 每条链保留 2000 → 共 8000 后验样本
TARGET_ACCEPT = 0.95   # NUTS 目标接受率，0.95 比默认 0.8 更稳

np.random.seed(RANDOM_SEED)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """配置并返回项目统一 logger，避免重复添加 handler。"""
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
