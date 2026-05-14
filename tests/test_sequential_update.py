"""
test_sequential_update.py — Module D 单元测试
测试切块逻辑和先验参数构造，不跑MCMC，速度很快。
用法：pytest tests/
"""

import pandas as pd
import numpy as np
import pytest

from src.config import (
    N_CHUNKS, CHUNK_SIZE,
    BETA_PRIOR_MEAN, BETA_PRIOR_SD, SIGMA_PRIOR_SCALE
)
from src.sequential_update import split_into_chunks, make_prior_params


def make_dummy_df(n: int = 360) -> pd.DataFrame:
    """生成测试用假数据，模拟day_subsampled.csv结构。"""
    np.random.seed(42)
    return pd.DataFrame({
        "dteday": pd.date_range("2011-01-01", periods=n, freq="D"),
        "cnt":    np.random.randint(500, 8000, size=n),
        "temp":   np.random.uniform(0, 1, size=n),
        "hum":    np.random.uniform(0, 1, size=n),
    })


# ── 切块测试 ─────────────────────────────────────────────────────────────────

class TestSplitIntoChunks:

    def test_correct_number_of_chunks(self):
        """切块数量必须等于N_CHUNKS=4。"""
        chunks = split_into_chunks(make_dummy_df())
        assert len(chunks) == N_CHUNKS

    def test_correct_chunk_size(self):
        """每块必须恰好CHUNK_SIZE=90行。"""
        chunks = split_into_chunks(make_dummy_df())
        for chunk in chunks:
            assert len(chunk) == CHUNK_SIZE

    def test_chunks_cover_all_rows(self):
        """4块合计必须覆盖全部360行，不多不少。"""
        df = make_dummy_df()
        chunks = split_into_chunks(df)
        assert sum(len(c) for c in chunks) == len(df)

    def test_chunks_are_chronological(self):
        """每块的最后日期必须早于下一块的第一个日期。"""
        chunks = split_into_chunks(make_dummy_df())
        for i in range(len(chunks) - 1):
            assert chunks[i]["dteday"].max() < chunks[i+1]["dteday"].min()

    def test_no_row_overlap(self):
        """任意两块之间不能有重叠行。"""
        chunks = split_into_chunks(make_dummy_df())
        indices = [set(c.index) for c in chunks]
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                assert indices[i].isdisjoint(indices[j])


# ── 先验参数测试 ──────────────────────────────────────────────────────────────

class TestMakePriorParams:

    def test_chunk0_uses_default_priors(self):
        """第1块必须使用B模块的弱信息先验。"""
        params = make_prior_params(chunk_idx=0, prev_posterior=None)
        assert params["beta0"] == (BETA_PRIOR_MEAN, BETA_PRIOR_SD)
        assert params["beta1"] == (BETA_PRIOR_MEAN, BETA_PRIOR_SD)
        assert params["beta2"] == (BETA_PRIOR_MEAN, BETA_PRIOR_SD)
        assert params["sigma_scale"] == 3000.0

    def test_chunk1_inherits_prev_posterior(self):
        """第2块的先验必须来自第1块的后验mean/sd。"""
        prev = {
            "beta0": {"mean": 2500.0, "sd": 380.0},
            "beta1": {"mean": 6800.0, "sd": 440.0},
            "beta2": {"mean": -2300.0, "sd": 560.0},
            "sigma": {"mean": 1500.0, "sd": 60.0},
        }
        params = make_prior_params(chunk_idx=1, prev_posterior=prev)
        assert params["beta0"] == (2500.0, 380.0)
        assert params["beta1"] == (6800.0, 440.0)
        assert params["beta2"] == (-2300.0, 560.0)
        assert params["sigma_scale"] == pytest.approx(1500.0 / np.sqrt(2 / np.pi))
        
    def test_sigma_scale_always_positive(self):
        """sigma_scale必须恒正，HalfNormal不接受负数。"""
        prev = {
            "beta0": {"mean": 0.0, "sd": 1.0},
            "beta1": {"mean": 0.0, "sd": 1.0},
            "beta2": {"mean": 0.0, "sd": 1.0},
            "sigma": {"mean": 500.0, "sd": 30.0},
        }
        params = make_prior_params(chunk_idx=2, prev_posterior=prev)
        assert params["sigma_scale"] > 0

    def test_none_posterior_triggers_default(self):
        """即使chunk_idx>0但prev_posterior=None，也应使用默认先验。"""
        params = make_prior_params(chunk_idx=2, prev_posterior=None)
        assert params["beta0"] == (BETA_PRIOR_MEAN, BETA_PRIOR_SD)