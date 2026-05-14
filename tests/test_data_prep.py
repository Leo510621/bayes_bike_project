from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.data_prep import subsample_data, validate_data


@pytest.fixture
def fake_raw_df() -> pd.DataFrame:
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
    result = subsample_data(fake_raw_df, n=360, random_state=42)
    assert len(result) == 360


def test_subsample_reproducible(fake_raw_df: pd.DataFrame) -> None:
    r1 = subsample_data(fake_raw_df, n=360, random_state=42)
    r2 = subsample_data(fake_raw_df, n=360, random_state=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_no_missing_values(fake_raw_df: pd.DataFrame) -> None:
    result = subsample_data(fake_raw_df, n=360, random_state=42)
    assert result.notna().all().all()


def test_sorted_by_dteday(fake_raw_df: pd.DataFrame) -> None:
    result = subsample_data(fake_raw_df, n=360, random_state=42)
    dates = result["dteday"].tolist()
    assert dates == sorted(dates)
