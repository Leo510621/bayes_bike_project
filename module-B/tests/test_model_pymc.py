"""
test_model_pymc.py — Module B 模型代码的快速单元测试

不跑完整 MCMC（耗时数分钟），只验证：
- 先验超参数符合预期数值（防误改）
- define_priors 在 PyMC 上下文中能正确创建 4 个变量
- build_model 在合成数据上能成功构建模型对象
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from src.priors import (
    BETA_PRIOR_MEAN,
    BETA_PRIOR_SD,
    SIGMA_PRIOR_SCALE,
    define_priors,
)



# 先验超参数硬编码检查


def test_prior_hyperparameters() -> None:
    """先验超参数应符合 §2.2 报告中声明的数值。"""
    assert BETA_PRIOR_MEAN == 0.0
    assert BETA_PRIOR_SD == 10000.0
    assert SIGMA_PRIOR_SCALE == 3000.0



# define_priors 行为


def test_define_priors_returns_four_variables() -> None:
    """先验函数应返回 4 个变量。"""
    with pm.Model():
        priors = define_priors()
    assert set(priors.keys()) == {"beta0", "beta1", "beta2", "sigma"}



# build_model 行为

@pytest.fixture
def fake_data():
    """构造合成数据用于模型构建测试。"""
    rng = np.random.default_rng(0)
    n = 100
    return {
        "cnt":  rng.uniform(0, 9000, n),
        "temp": rng.uniform(0, 1, n),
        "hum":  rng.uniform(0, 1, n),
    }


def test_build_model_returns_pymc_model(fake_data) -> None:
    """build_model 应返回 pm.Model 对象，且包含 4 个未观测随机变量。"""
    from src.model_pymc import build_model

    model = build_model(fake_data["cnt"], fake_data["temp"], fake_data["hum"])
    assert isinstance(model, pm.Model)

    free_var_names = {rv.name for rv in model.free_RVs}
    assert {"beta0", "beta1", "beta2", "sigma"} <= free_var_names
