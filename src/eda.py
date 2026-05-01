"""
eda.py — 探索性数据分析与过度离散检验

对子采样数据执行描述性统计、绘制直方图/散点图/热力图，
并计算 Var/Mean ratio 以论证 Poisson 不适用、应采用 Normal 或 NegBin。
所有图表保存至 results/figures/，数值结果保存至 results/。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import (
    DATA_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    SUBSAMPLED_FILE,
    setup_logging,
)


def load_subsampled(filepath: pd.io.common.PathLike) -> pd.DataFrame:
    """加载子采样 CSV 文件。"""
    return pd.read_csv(filepath, parse_dates=["dteday"])


def summary_statistics(df: pd.DataFrame, save_path: pd.io.common.PathLike) -> pd.DataFrame:
    """计算 cnt/temp/hum 的描述性统计并保存为 CSV。"""
    stats_df = df[["cnt", "temp", "hum"]].describe(
        percentiles=[0.25, 0.5, 0.75]
    ).T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_df.columns = ["mean", "std", "min", "q25", "q50", "q75", "max"]
    stats_df.to_csv(save_path)
    return stats_df


def plot_cnt_histogram(df: pd.DataFrame, save_path: pd.io.common.PathLike) -> None:
    """绘制 cnt 直方图 + KDE 密度曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["cnt"].values

    ax.hist(counts, bins=30, density=True, alpha=0.7, edgecolor="black", label="Histogram")

    kde = stats.gaussian_kde(counts)
    x_grid = np.linspace(counts.min(), counts.max(), 200)
    ax.plot(x_grid, kde(x_grid), linewidth=2, color="red", label="KDE")

    ax.set_xlabel("Daily Rental Count (cnt)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Daily Bike Rentals (n=360)")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_with_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: pd.io.common.PathLike,
    title: str | None = None,
) -> None:
    """绘制散点图 + 线性拟合线，标注斜率和 R²。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df[x_col].values
    y = df[y_col].values

    ax.scatter(x, y, alpha=0.5, s=20)

    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    r_squared = r ** 2

    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=2)

    ax.annotate(
        f"slope = {slope:.1f}\nR² = {r_squared:.3f}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs {x_col}")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: pd.io.common.PathLike) -> None:
    """绘制所有数值列的相关系数热力图。"""
    numeric_df = df.select_dtypes("number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Subsampled Data)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_overdispersion_ratio(df: pd.DataFrame, save_path: pd.io.common.PathLike) -> dict:
    """计算 cnt 的 Var/Mean ratio，写入诊断文本文件。"""
    cnt = df["cnt"]
    mean_cnt = cnt.mean()
    var_cnt = cnt.var()
    ratio = var_cnt / mean_cnt

    result = {
        "mean": mean_cnt,
        "variance": var_cnt,
        "ratio": ratio,
    }

    lines = [
        f"mean(cnt) = {mean_cnt:.2f}",
        f"var(cnt)  = {var_cnt:.2f}",
        f"Var/Mean ratio = {ratio:.2f}",
        "",
        "Conclusion: Severe overdispersion (ratio >> 1). "
        "Poisson likelihood is inappropriate; we adopt Normal as primary "
        "and discuss Negative Binomial as alternative.",
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    return result


def main() -> None:
    """执行完整 EDA 流程：统计 → 直方图 → 散点图 → 热力图 → 过度离散检验。"""
    logger = setup_logging()

    logger.info("Loading subsampled data")
    df = load_subsampled(DATA_DIR / SUBSAMPLED_FILE)

    logger.info("Computing summary statistics")
    summary_statistics(df, RESULTS_DIR / "eda_summary_stats.csv")

    logger.info("Plotting cnt histogram")
    plot_cnt_histogram(df, FIGURES_DIR / "cnt_histogram.png")

    logger.info("Plotting cnt vs temp scatter")
    plot_scatter_with_fit(
        df, "temp", "cnt",
        FIGURES_DIR / "cnt_vs_temp_scatter.png",
        title="Daily Rentals vs Temperature",
    )

    logger.info("Plotting cnt vs hum scatter")
    plot_scatter_with_fit(
        df, "hum", "cnt",
        FIGURES_DIR / "cnt_vs_hum_scatter.png",
        title="Daily Rentals vs Humidity",
    )

    logger.info("Plotting correlation heatmap")
    plot_correlation_heatmap(df, FIGURES_DIR / "correlation_heatmap.png")

    logger.info("Computing overdispersion ratio")
    compute_overdispersion_ratio(df, RESULTS_DIR / "overdispersion_check.txt")

    logger.info("EDA complete")


if __name__ == "__main__":
    main()
