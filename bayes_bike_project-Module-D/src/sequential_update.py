"""
sequential_update.py — Extension C: Sequential Bayesian Updating

流程：把360行数据按时间切成4块（每块90行），
用上一块的后验作为下一块的先验，级联更新。

用法：
    python -m src.sequential_update
"""

from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pymc as pm

from src.config import (
    BETA_PRIOR_MEAN,
    BETA_PRIOR_SD,
    CHUNK_SIZE,
    DATA_DIR,
    FEATURES,
    FIGURES_DIR,
    N_CHAINS,
    N_CHUNKS,
    N_DRAWS,
    N_TUNE,
    RANDOM_SEED,
    RESULTS_DIR,
    SEQUENTIAL_TABLE_FILE,
    SIGMA_PRIOR_SCALE,
    SUBSAMPLED_FILE,
    TARGET,
    TARGET_ACCEPT,
    setup_logging,
)


def load_data() -> pd.DataFrame:
    """加载A模块生成的子采样数据（已按日期升序排列）。"""
    path = DATA_DIR / SUBSAMPLED_FILE
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请确认 data/ 目录下有 day_subsampled.csv")
    return pd.read_csv(path, parse_dates=["dteday"])

def split_into_chunks(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    将数据按时间顺序均匀切成 N_CHUNKS 块。
    
    块0: 行 0-89   （约2011年前期）
    块1: 行 90-179 （约2011年后期）
    块2: 行 180-269（约2012年前期）
    块3: 行 270-359（约2012年后期）
    """
    return [
        df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE].copy()
        for i in range(N_CHUNKS)
    ]

def make_prior_params(chunk_idx: int, prev_posterior: dict | None) -> dict:
    """
    返回当前块的先验超参数。

    chunk_idx == 0:
        用B模块的弱信息先验（第一块没有"上一块"）
    chunk_idx > 0:
        矩匹配——把上一块后验的 mean/sd 直接当作本块的先验 mu/sigma

    Parameters
    ----------
    chunk_idx      : 当前是第几块（0, 1, 2, 3）
    prev_posterior : 上一块的后验摘要，格式：
                     {"beta0": {"mean": ..., "sd": ...}, ...}
                     第一块传入 None

    Returns
    -------
    dict: {"beta0": (mu, sd), "beta1": (mu, sd),
           "beta2": (mu, sd), "sigma_scale": float}
    """
    if chunk_idx == 0 or prev_posterior is None:
        return {
            "beta0": (BETA_PRIOR_MEAN, BETA_PRIOR_SD),
            "beta1": (BETA_PRIOR_MEAN, BETA_PRIOR_SD),
            "beta2": (BETA_PRIOR_MEAN, BETA_PRIOR_SD),
            "sigma_scale": SIGMA_PRIOR_SCALE,
        }
    else:
        return {
            "beta0": (prev_posterior["beta0"]["mean"], prev_posterior["beta0"]["sd"]),
            "beta1": (prev_posterior["beta1"]["mean"], prev_posterior["beta1"]["sd"]),
            "beta2": (prev_posterior["beta2"]["mean"], prev_posterior["beta2"]["sd"]),
            "sigma_scale": prev_posterior["sigma"]["mean"],
        }
    

def build_and_sample(
    chunk: pd.DataFrame,
    prior_params: dict,
    chunk_idx: int,
) -> az.InferenceData:
    """
    针对单块数据构建PyMC模型并运行NUTS采样。

    Parameters
    ----------
    chunk        : 当前块的DataFrame（90行）
    prior_params : make_prior_params()返回的先验超参数
    chunk_idx    : 块编号，用于设定随机种子

    Returns
    -------
    ArviZ InferenceData（含后验样本）
    """
    cnt  = chunk[TARGET].to_numpy()
    temp = chunk[FEATURES[0]].to_numpy()
    hum  = chunk[FEATURES[1]].to_numpy()

    b0_mu, b0_sd = prior_params["beta0"]
    b1_mu, b1_sd = prior_params["beta1"]
    b2_mu, b2_sd = prior_params["beta2"]
    sigma_scale   = prior_params["sigma_scale"]

    with pm.Model():
        # 先验
        beta0 = pm.Normal("beta0", mu=b0_mu, sigma=b0_sd)
        beta1 = pm.Normal("beta1", mu=b1_mu, sigma=b1_sd)
        beta2 = pm.Normal("beta2", mu=b2_mu, sigma=b2_sd)
        sigma = pm.HalfNormal("sigma", sigma=sigma_scale)

        # 线性预测子
        mu = beta0 + beta1 * temp + beta2 * hum

        # 似然
        pm.Normal("cnt_obs", mu=mu, sigma=sigma, observed=cnt)

        # 采样
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED + chunk_idx,
            return_inferencedata=True,
            progressbar=True,
        )

    return idata


def extract_posterior_summary(idata: az.InferenceData) -> dict:
    """
    从InferenceData提取4个参数的后验 mean/sd/95% CrI。

    Returns
    -------
    dict: {"beta0": {"mean":..., "sd":..., "lower":..., "upper":...,
                     "r_hat":..., "ess":...}, ...}
    """
    summary = az.summary(
        idata,
        var_names=["beta0", "beta1", "beta2", "sigma"],
        hdi_prob=0.95,
    )

    result = {}
    for var in ["beta0", "beta1", "beta2", "sigma"]:
        result[var] = {
            "mean":  float(summary.loc[var, "mean"]),
            "sd":    float(summary.loc[var, "sd"]),
            "lower": float(summary.loc[var, "hdi_2.5%"]),
            "upper": float(summary.loc[var, "hdi_97.5%"]),
            "r_hat": float(summary.loc[var, "r_hat"]),
            "ess":   float(summary.loc[var, "ess_bulk"]),
        }
    return result

PARAM_META = {
    "beta0": {"label": r"$\beta_0$ (intercept)", "color": "#2E86AB"},
    "beta1": {"label": r"$\beta_1$ (temp)",      "color": "#E07A5F"},
    "beta2": {"label": r"$\beta_2$ (hum)",       "color": "#81B29A"},
    "sigma": {"label": r"$\sigma$",              "color": "#9D4EDD"},
}


def plot_cascade(all_summaries: list[dict]) -> None:
    """
    Cascade Plot：4个参数 × 4个块的后验均值 + 95% CrI。
    竖线随块数增加越来越窄，展示不确定性的收缩。
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    axes_flat = axes.flatten()
    x = np.arange(1, N_CHUNKS + 1)
    alphas = [0.35, 0.55, 0.75, 1.00]
    chunk_labels = [f"Chunk {i+1}\n(rows {i*90+1}–{(i+1)*90})" for i in range(N_CHUNKS)]

    for ax, param in zip(axes_flat, PARAM_META):
        meta  = PARAM_META[param]
        color = meta["color"]

        means  = [s[param]["mean"]  for s in all_summaries]
        lowers = [s[param]["lower"] for s in all_summaries]
        uppers = [s[param]["upper"] for s in all_summaries]

        for i in range(N_CHUNKS):
            # 95% CrI 竖线
            ax.plot(
                [x[i], x[i]], [lowers[i], uppers[i]],
                color=color, linewidth=3,
                alpha=alphas[i], solid_capstyle="round",
            )
            # 后验均值点
            ax.scatter(
                x[i], means[i], s=80, color=color,
                alpha=alphas[i], zorder=5,
                edgecolors="white", linewidths=1.5,
            )
            # 数值标注
            ax.text(
                x[i], uppers[i] + (uppers[i] - lowers[i]) * 0.05,
                f"{means[i]:.0f}",
                ha="center", va="bottom",
                fontsize=9, color=color, alpha=alphas[i],
            )

        # 均值连线
        ax.plot(x, means, color=color, linewidth=1.2,
                linestyle="--", alpha=0.6)

        ax.set_title(meta["label"], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(chunk_labels, fontsize=10)
        ax.set_ylabel("Parameter value", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#FAFAFA")

    fig.suptitle(
        "Sequential Bayesian Updating — Posterior Evolution Across Chunks",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cascade_plot.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_density_evolution(all_idatas: list[az.InferenceData]) -> None:
    """
    Density Evolution：把4块的后验密度曲线叠加在一起。
    由浅到深 = 块1到块4，直观展示分布从宽胖变窄高的过程。
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    axes_flat = axes.flatten()
    alphas = [0.35, 0.55, 0.75, 1.00]
    labels = [f"Chunk {i+1} (rows {i*90+1}–{(i+1)*90})" for i in range(N_CHUNKS)]

    for ax, param in zip(axes_flat, PARAM_META):
        meta  = PARAM_META[param]
        color = meta["color"]

        for i, idata in enumerate(all_idatas):
            # 把4条链的样本合并成一个一维数组
            samples = idata.posterior[param].values.flatten()

            ax.hist(
                samples, bins=60, density=True,
                color=color, alpha=alphas[i],
                label=labels[i], edgecolor="none",
            )

        ax.set_title(meta["label"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Parameter value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#FAFAFA")

    fig.suptitle(
        "Posterior Density Evolution — Bayesian Learning Across Chunks",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "density_evolution.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

def save_summary_table(all_summaries: list[dict]) -> None:
    """把4块的后验摘要整合成CSV，方便填写section6的TODO表格。"""
    rows = []
    param_alias = {
        "beta0": "beta_0",
        "beta1": "beta_1_temp",
        "beta2": "beta_2_hum",
        "sigma": "sigma",
    }
    for i, summary in enumerate(all_summaries):
        for var, alias in param_alias.items():
            s = summary[var]
            rows.append({
                "chunk":     i + 1,
                "rows": f"{i*90+1}-{(i+1)*90}",
                "parameter": alias,
                "mean":      round(s["mean"],  2),
                "sd":        round(s["sd"],    2),
                "ci_lower":  round(s["lower"], 2),
                "ci_upper":  round(s["upper"], 2),
                "r_hat":     round(s["r_hat"], 3),
                "ess":       int(s["ess"]),
            })
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / SEQUENTIAL_TABLE_FILE, index=False
    )


def main() -> None:
    logger = setup_logging()
    logger.info("=== Module D: Sequential Bayesian Updating ===")

    # 1. 加载数据并切块
    df     = load_data()
    chunks = split_into_chunks(df)
    logger.info("Data chunks finished：%d chunks，each chunk %d rows", N_CHUNKS, CHUNK_SIZE)

    # 2. 逐块采样（核心循环）
    all_summaries = []
    all_idatas    = []
    prev_posterior = None

    for i, chunk in enumerate(chunks):
        logger.info("── chunk %d / %d ──", i + 1, N_CHUNKS)

        prior_params = make_prior_params(i, prev_posterior)
        idata        = build_and_sample(chunk, prior_params, chunk_idx=i)
        summary      = extract_posterior_summary(idata)

        all_summaries.append(summary)
        all_idatas.append(idata)
        prev_posterior = summary

        logger.info(
            "β₁ posterior: mean=%.0f, 95%%CrI=[%.0f, %.0f], R̂=%.3f",
            summary["beta1"]["mean"],
            summary["beta1"]["lower"],
            summary["beta1"]["upper"],
            summary["beta1"]["r_hat"],
        )

    # 3. 画图
    logger.info("draw Cascade Plot")
    plot_cascade(all_summaries)

    logger.info("draw Density Evolution Plot")
    plot_density_evolution(all_idatas)

    # 4. 保存汇总表
    save_summary_table(all_summaries)
    logger.info("results had been saved to results/")
    logger.info("finished.")


if __name__ == "__main__":
    main()