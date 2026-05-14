"""
priors.py — 弱信息先验定义

为贝叶斯线性回归
    cnt_i ~ N(beta0 + beta1 * temp_i + beta2 * hum_i, sigma^2)
指定 4 个未知参数的先验分布。

设计原则（QHM §3.2）：弱信息先验
  - 排除明显荒谬的取值
  - 不强烈偏向任何特定值
  - 不强行压缩后验，让数据主导

超参数选择依据（来源：Yue的 results/eda_summary_stats.csv）：
  - SD(cnt) ≈ 1936，预测变量 temp/hum ∈ [0, 1]
  - OLS 估计：beta0 ≈ 2548, beta1 ≈ 6796, beta2 ≈ -2370, sigma_hat ≈ 1497
  - 真实参数量级在数千，所以先验离散度需在 10^4 量级才算"弱"
"""

from __future__ import annotations

import pymc as pm


# 先验超参数


# 回归系数 beta0, beta1, beta2 的先验：N(mean=0, sd=10000)
#   - mean = 0：不预先假设效应方向（让数据决定 beta 是正还是负）
#   - sd   = 10000：95% 先验区间 ±20000，覆盖一切合理值
#                   排除荒谬值（|beta| > 5e4 在数据尺度下不可能）
#                   相对于真实 beta（千量级），这个先验几乎是平的
BETA_PRIOR_MEAN = 0.0
BETA_PRIOR_SD = 10000.0

# 残差标准差 sigma 的先验：HalfNormal(scale=3000)
#   - 必须 > 0（标准差不能为负），HalfNormal 自动满足
#   - scale = 3000：先验中位数 ≈ 2024，95 分位 ≈ 5880
#   - 与 SD(cnt) ≈ 1936 同量级，覆盖 OLS 残差 sigma_hat = 1497
#   - 排除 sigma > 10000 的不合理值
SIGMA_PRIOR_SCALE = 3000.0



# 先验定义函数


def define_priors() -> dict:
    """
    在 PyMC 模型上下文中创建所有先验随机变量。

    必须在 `with pm.Model() as model:` 块内调用。

    Returns
    -------
    dict
        参数名 -> PyMC 随机变量。
        键: "beta0", "beta1", "beta2", "sigma"
    """
    beta0 = pm.Normal("beta0", mu=BETA_PRIOR_MEAN, sigma=BETA_PRIOR_SD)
    beta1 = pm.Normal("beta1", mu=BETA_PRIOR_MEAN, sigma=BETA_PRIOR_SD)
    beta2 = pm.Normal("beta2", mu=BETA_PRIOR_MEAN, sigma=BETA_PRIOR_SD)
    sigma = pm.HalfNormal("sigma", sigma=SIGMA_PRIOR_SCALE)
    return {"beta0": beta0, "beta1": beta1, "beta2": beta2, "sigma": sigma}


def prior_summary() -> str:
    """
    返回先验设定的可读摘要（写入报告 §2.2 时直接复制）。
    """
    return (
        f"beta0 ~ Normal(mu={BETA_PRIOR_MEAN}, sigma={BETA_PRIOR_SD})\n"
        f"beta1 ~ Normal(mu={BETA_PRIOR_MEAN}, sigma={BETA_PRIOR_SD})\n"
        f"beta2 ~ Normal(mu={BETA_PRIOR_MEAN}, sigma={BETA_PRIOR_SD})\n"
        f"sigma ~ HalfNormal(sigma={SIGMA_PRIOR_SCALE})"
    )


if __name__ == "__main__":
    # 在终端打印一下设定，便于调试
    print(prior_summary())
