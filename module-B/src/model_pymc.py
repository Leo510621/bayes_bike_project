"""
model_pymc.py — 贝叶斯线性回归 PyMC 模型 + MCMC 采样

模型规格:
    cnt_i ~ N(mu_i, sigma^2)
    mu_i  = beta0 + beta1 * temp_i + beta2 * hum_i
    i = 1, ..., 360

采样器: NUTS (PyMC 默认，No-U-Turn Sampler)
    - 4 chains × (1000 tune + 2000 draws) = 8000 后验样本
    - tune 阶段同时担任 warm-up 和 burn-in，自动调步长后丢弃
    - random_seed=42 保证可复现

输入: data/day_subsampled.csv (来自 A 模块)
输出:
    results/trace.nc                       # 后验样本 (供 C/D)
    results/figures/prior_predictive.png   # 先验预测检查
"""

from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from src.config import (
    DATA_DIR,
    FIGURES_DIR,
    FEATURES,
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    POSTERIOR_TABLE_FILE,  # noqa: F401  (不在本文件用，但有助于 IDE 跳转)
    RANDOM_SEED,
    RESULTS_DIR,
    SUBSAMPLED_FILE,
    TARGET,
    TARGET_ACCEPT,
    TRACE_FILE,
    setup_logging,
)
from src.priors import define_priors, prior_summary



# 数据加载

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 Yue 生成的 day_subsampled.csv 中读取 cnt, temp, hum 三列。

    Returns
    -------
    cnt : np.ndarray, shape (360,)
    temp: np.ndarray, shape (360,)
    hum : np.ndarray, shape (360,)
    """
    csv_path = DATA_DIR / SUBSAMPLED_FILE
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到 {csv_path}。\n"
            "请先从 Yue 的项目复制 day_subsampled.csv 到 data/ 目录。"
        )
    df = pd.read_csv(csv_path)
    return (
        df[TARGET].to_numpy(),
        df[FEATURES[0]].to_numpy(),
        df[FEATURES[1]].to_numpy(),
    )



# 模型构建

def build_model(
    cnt: np.ndarray, temp: np.ndarray, hum: np.ndarray
) -> pm.Model:
    """
    构建 PyMC 模型对象（先验 + 似然），不执行采样。

    Parameters
    ----------
    cnt, temp, hum : 来自 day_subsampled.csv 的 360 维数组
    """
    with pm.Model() as model:
        # 先验（来自 priors.py）
        priors = define_priors()

        # 线性预测子 mu_i = beta0 + beta1 * temp_i + beta2 * hum_i
        mu = (
            priors["beta0"]
            + priors["beta1"] * temp
            + priors["beta2"] * hum
        )

        # 似然 cnt_i ~ N(mu_i, sigma^2)
        #    observed=cnt 把数据告诉 PyMC，让它自动计算似然
        pm.Normal("cnt_obs", mu=mu, sigma=priors["sigma"], observed=cnt)

    return model



# 先验预测检查


def run_prior_predictive(model: pm.Model, save_path) -> None:
    """
    从先验预测分布中采样 500 份合成数据，画分布图。

    目的：在见数据前验证先验是否"合理地宽"——
    如果先验预测分布远窄于实际 cnt 范围 → 先验太紧；
    如果远宽到含极端值 → 没问题，弱信息先验本就允许。
    """
    with model:
        prior_pred = pm.sample_prior_predictive(
            samples=500, random_seed=RANDOM_SEED
        )

    samples = prior_pred.prior_predictive["cnt_obs"].values.flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        samples, bins=80, density=True, alpha=0.6,
        edgecolor="black", label="Prior predictive draws",
    )
    ax.axvline(0, color="red", linestyle="--", label="cnt = 0")
    ax.axvline(8555, color="green", linestyle="--", label="Observed max cnt")
    ax.set_xlabel("Simulated cnt")
    ax.set_ylabel("Density")
    ax.set_title("Prior Predictive Distribution (n=500)")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# MCMC 采样


def run_mcmc(model: pm.Model) -> az.InferenceData:
    """
    在已构建的模型上跑 NUTS 采样。

    返回 ArviZ InferenceData，可直接保存为 .nc 文件。
    """
    with model:
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            progressbar=True,
        )
    return idata



# 主流程


def main() -> None:
    """完整流程：加载数据 → 建模 → 先验预测 → MCMC → 保存 trace.nc"""
    logger = setup_logging()

    logger.info("=== Module B: Bayesian Inference ===")
    logger.info("Prior specification:\n%s", prior_summary())

    logger.info("Loading subsampled data")
    cnt, temp, hum = load_data()
    logger.info(
        "Data loaded: n=%d, mean(cnt)=%.1f, sd(cnt)=%.1f",
        len(cnt), cnt.mean(), cnt.std(),
    )

    logger.info("Building PyMC model")
    model = build_model(cnt, temp, hum)

    logger.info("Running prior predictive check (500 samples)")
    run_prior_predictive(model, FIGURES_DIR / "prior_predictive.png")

    logger.info(
        "Running NUTS sampling: %d chains × (%d tune + %d draws)",
        N_CHAINS, N_TUNE, N_DRAWS,
    )
    idata = run_mcmc(model)

    trace_path = RESULTS_DIR / TRACE_FILE
    idata.to_netcdf(trace_path)
    logger.info("Saved trace to %s", trace_path)

    # 打印简易摘要
    summary = az.summary(
        idata,
        var_names=["beta0", "beta1", "beta2", "sigma"],
        hdi_prob=0.95,
    )
    logger.info("Quick posterior summary:\n%s", summary.to_string())

    logger.info("Done. Run posterior_summary.py next.")


if __name__ == "__main__":
    main()
