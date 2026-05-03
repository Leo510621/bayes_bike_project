"""
posterior_summary.py — 后验摘要表 + 后验密度图 + 森林图

读取 trace.nc，输出：
  results/posterior_table.csv          # 后验摘要 (mean/sd/95% CrI/ESS/R-hat)
  results/figures/posterior_density.png # 4 个参数的边际后验密度
  results/figures/posterior_forest.png  # 森林图：可信区间一目了然

字段命名遵循 Module A 的 frequentist_results.csv 契约：
    parameter ∈ {beta_0, beta_1_temp, beta_2_hum, sigma}
便于 D 同学整合时直接 merge。
"""

from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    POSTERIOR_TABLE_FILE,
    RESULTS_DIR,
    TRACE_FILE,
    setup_logging,
)



# 读 trace


def load_trace() -> az.InferenceData:
    """读取 model_pymc.py 生成的 trace.nc 文件。"""
    trace_path = RESULTS_DIR / TRACE_FILE
    if not trace_path.exists():
        raise FileNotFoundError(
            f"未找到 {trace_path}。请先运行 `python -m src.model_pymc`。"
        )
    return az.from_netcdf(trace_path)



# 摘要表


def build_summary_table(idata: az.InferenceData) -> pd.DataFrame:
    """
    生成符合 Module A 契约的后验摘要表。

    列名: parameter, post_mean, post_sd, ci_lower, ci_upper, ess_bulk, r_hat
    参数名: beta_0, beta_1_temp, beta_2_hum, sigma  (与 A 完全一致)
    """
    var_names = ["beta0", "beta1", "beta2", "sigma"]
    parameter_aliases = ["beta_0", "beta_1_temp", "beta_2_hum", "sigma"]

    summary = az.summary(
        idata,
        var_names=var_names,
        hdi_prob=0.95,
        kind="all",
    )

    # 注意：az.summary 的 HDI 列在不同版本里命名可能是
    #   "hdi_2.5%" / "hdi_97.5%" 或 "hdi_lower" / "hdi_upper"
    hdi_lower_col = next(c for c in summary.columns if "hdi" in c and ("2.5" in c or "lower" in c))
    hdi_upper_col = next(c for c in summary.columns if "hdi" in c and ("97.5" in c or "upper" in c))

    df = pd.DataFrame({
        "parameter": parameter_aliases,
        "post_mean": summary["mean"].values,
        "post_sd":   summary["sd"].values,
        "ci_lower":  summary[hdi_lower_col].values,
        "ci_upper":  summary[hdi_upper_col].values,
        "ess_bulk":  summary["ess_bulk"].values,
        "r_hat":     summary["r_hat"].values,
    })
    return df


# 后验密度图


# 4 个参数对应的颜色（柔和但区分度高的配色）
PARAM_COLORS = {
    "beta0": "#2E86AB",   # 海蓝
    "beta1": "#E07A5F",   # 砖红
    "beta2": "#81B29A",   # 薄荷绿
    "sigma": "#9D4EDD",   # 紫
}
PARAM_LABELS = {
    "beta0": r"$\beta_0$ (intercept)",
    "beta1": r"$\beta_1$ (temp)",
    "beta2": r"$\beta_2$ (hum)",
    "sigma": r"$\sigma$",
}


def plot_posterior_density(idata: az.InferenceData, save_path) -> None:
    """绘制 4 个参数的边际后验密度图 (2x2 子图)，每个子图独立配色。"""
    var_names = list(PARAM_COLORS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor="white")
    axes_flat = axes.flatten()

    for ax, var in zip(axes_flat, var_names):
        az.plot_posterior(
            idata,
            var_names=[var],
            hdi_prob=0.95,
            ax=ax,
            color=PARAM_COLORS[var],
            point_estimate="mean",
            textsize=11,
        )
        # 用更友好的标题替换原 var name
        ax.set_title(PARAM_LABELS[var], fontsize=13, weight="bold")
        # 加一层柔和的网格
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        ax.set_facecolor("#FAFAFA")

    fig.suptitle(
        "Posterior Distributions of Parameters",
        fontsize=15, weight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_posterior_forest(idata: az.InferenceData, save_path) -> None:
    """
    自定义森林图：4 个参数的 95% CrI，每个参数:
      - 蓝色粗线表示 HDI 区间
      - 白圆点 = 后验均值
      - 三条红色虚线：lower / mean / upper 各一条，落到 x 轴
      - 三个数值文本标在区间上方
    """
    var_names = list(PARAM_COLORS.keys())

    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95)
    hdi_lower_col = next(
        c for c in summary.columns if "hdi" in c and ("2.5" in c or "lower" in c)
    )
    hdi_upper_col = next(
        c for c in summary.columns if "hdi" in c and ("97.5" in c or "upper" in c)
    )

    means = summary["mean"].values
    lowers = summary[hdi_lower_col].values
    uppers = summary[hdi_upper_col].values

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")

    n = len(var_names)
    y_positions = np.arange(n)[::-1]   # 让第一个参数显示在最上方

    # 红色竖线落到 y 轴这个位置（数据坐标）
    y_baseline = -0.6

    for y, var, mean, lower, upper in zip(
        y_positions, var_names, means, lowers, uppers
    ):
        bar_color = PARAM_COLORS[var]

        # HDI 区间（粗实线）
        ax.plot(
            [lower, upper], [y, y],
            color=bar_color, linewidth=6,
            solid_capstyle="round", zorder=3,
        )

        # 三条红色虚线：lower / mean / upper，均落到 y_baseline
        for x_val in [lower, mean, upper]:
            ax.plot(
                [x_val, x_val], [y, y_baseline],
                color="red", linestyle="--", linewidth=1.2,
                alpha=0.55, zorder=2,
            )

        # 数值文本（区间上方）
        # sigma 的 CrI 很窄（1393-1610），左右标签需向区间外侧错开避免重叠
        if var == "sigma":
            ax.text(lower - 120, y + 0.18, f"{lower:.0f}",
                    ha="right", va="bottom", fontsize=10, color="red")
            ax.text(upper + 120, y + 0.18, f"{upper:.0f}",
                    ha="left", va="bottom", fontsize=10, color="red")
        else:
            ax.text(lower, y + 0.18, f"{lower:.0f}",
                    ha="center", va="bottom", fontsize=10, color="red")
            ax.text(upper, y + 0.18, f"{upper:.0f}",
                    ha="center", va="bottom", fontsize=10, color="red")
        ax.text(mean, y + 0.30, f"mean = {mean:.0f}",
                ha="center", va="bottom", fontsize=11, color="black", weight="bold")

        # 后验均值标记（白心圆）
        ax.scatter(
            mean, y, s=140,
            facecolor="white", edgecolor=bar_color,
            linewidths=2.2, zorder=5,
        )

    # y 轴：参数名
    ax.set_yticks(y_positions)
    ax.set_yticklabels([PARAM_LABELS[v] for v in var_names], fontsize=12)
    ax.set_ylim(y_baseline - 0.2, n - 0.4)

    # x 轴
    ax.set_xlabel("Parameter value", fontsize=12)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, zorder=1)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # 标题
    ax.set_title(
        "Forest Plot: 95% Credible Intervals",
        fontsize=14, weight="bold", pad=15,
    )

    # 干净的边框
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 主流程


def main() -> None:
    logger = setup_logging()

    logger.info("Loading trace.nc")
    idata = load_trace()

    logger.info("Building posterior summary table")
    summary_df = build_summary_table(idata)
    summary_path = RESULTS_DIR / POSTERIOR_TABLE_FILE
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved posterior table to %s", summary_path)

    print("\n=== Posterior Summary ===")
    print(summary_df.to_string(index=False))
    print()

    logger.info("Plotting posterior density")
    plot_posterior_density(idata, FIGURES_DIR / "posterior_density.png")

    logger.info("Plotting forest plot")
    plot_posterior_forest(idata, FIGURES_DIR / "posterior_forest.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
