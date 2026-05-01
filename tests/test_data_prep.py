"""
test_data_prep.py — data_prep 模块的单元测试

使用合成数据（不依赖 day.csv）验证子采样逻辑：
行数正确、结果可复现、无缺失值、按日期排序。
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.data_prep import subsample_data, validate_data


@pytest.fixture
def fake_raw_df() -> pd.DataFrame:
    """构造 731 行合成 DataFrame，模拟 day.csv 结构。"""
    rng = np.random.default_rng(0)
    n = 731
    return pd.DataFrame({
        "dteday": pd.date_range("2011-01-01", periods=n, freq="D"),
        "cnt": rng.integers(100, 9000, size=n),
        "temp": rng.uniform(0, 1, size=n),
        "hum": rng.uniform(0, 1, size=n),
        "season": rng.integers(1, 5, size=n),
        "yr": np.concatenate([np.zeros(365), np.ones(366)]).astype(int),
        "mnth": np.tile(np.arange(1, 13), 61)[:n],
        "holiday": rng.integers(0, 2, size=n),
        "weekday": np.tile(np.arange(7), 105)[:n],
        "workingday": rng.integers(0, 2, size=n),
        "weathersit": rng.integers(1, 4, size=n),
        "atemp": rng.uniform(0, 1, size=n),
        "windspeed": rng.uniform(0, 1, size=n),
        "casual": rng.integers(0, 3000, size=n),
        "registered": rng.integers(0, 7000, size=n),
        "instant": np.arange(1, n + 1),
    })


def test_subsample_correct_size(fake_raw_df: pd.DataFrame) -> None:
    """子采样结果应恰好 360 行。"""
    result = subsample_data(fake_raw_df, n=360, random_state=42)
    assert len(result) == 360


def test_subsample_reproducible(fake_raw_df: pd.DataFrame) -> None:
    """相同 random_state 应产生完全相同的输出。"""
    r1 = subsample_data(fake_raw_df, n=360, random_state=42)
    r2 = subsample_data(fake_raw_df, n=360, random_state=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_no_missing_values(fake_raw_df: pd.DataFrame) -> None:
    """子采样结果不应包含任何 NaN。"""
    result = subsample_data(fake_raw_df, n=360, random_state=42)
    assert result.notna().all().all()


def test_sorted_by_dteday(fake_raw_df: pd.DataFrame) -> None:
    """子采样结果应按 dteday 升序排列。"""
    result = subsample_data(fake_raw_df, n=360, random_state=42)
    dates = result["dteday"].tolist()
    assert dates == sorted(dates)
