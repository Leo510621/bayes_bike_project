"""
frequentist_baseline.py — 频率派 OLS 基线模型

对子采样数据拟合 cnt ~ 1 + temp + hum 的普通最小二乘回归，
提取参数估计（MLE、标准误、95% CI、p 值）和模型诊断（R²、F 检验、BP 异方差检验），
分别保存至 frequentist_results.csv 和 frequentist_diagnostics.csv。
供 §3.2 频率派 vs 贝叶斯对比使用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from src.config import (
    DATA_DIR,
    FEATURES,
    RESULTS_DIR,
    SUBSAMPLED_FILE,
    TARGET,
    setup_logging,
)


def load_data(filepath: pd.io.common.PathLike) -> pd.DataFrame:
    """加载子采样 CSV 文件。"""
    return pd.read_csv(filepath)


def fit_ols(
    df: pd.DataFrame,
    target: str = TARGET,
    features: list[str] = FEATURES,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """拟合带截距的 OLS 模型：cnt ~ 1 + temp + hum。"""
    X = sm.add_constant(df[features])
    y = df[target]
    model = sm.OLS(y, X).fit()
    return model


def summarize_results(model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """提取参数估计，输出符合数据契约的 DataFrame。"""
    param_names = ["beta_0", "beta_1_temp", "beta_2_hum"]
    conf = model.conf_int()

    rows = []
    for i, name in enumerate(param_names):
        rows.append({
            "parameter": name,
            "mle": model.params.iloc[i],
            "std_err": model.bse.iloc[i],
            "ci_lower": conf.iloc[i, 0],
            "ci_upper": conf.iloc[i, 1],
            "p_value": model.pvalues.iloc[i],
        })

    sigma = np.sqrt(model.mse_resid)
    rows.append({
        "parameter": "sigma",
        "mle": sigma,
        "std_err": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
    })

    return pd.DataFrame(rows)


def extract_model_diagnostics(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> dict:
    """提取模型诊断指标：R²、F 检验、RMSE、Breusch-Pagan 异方差检验。"""
    resid = model.resid
    exog = model.model.exog

    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(resid, exog)

    return {
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "rmse": np.sqrt(model.mse_resid),
        "n_obs": int(model.nobs),
        "df_resid": int(model.df_resid),
        "bp_lm_stat": bp_lm,
        "bp_lm_pvalue": bp_lm_pvalue,
    }


def print_full_summary(model: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    """打印 statsmodels 完整 OLS 回归摘要。"""
    print(model.summary())


def main() -> None:
    """执行频率派基线流程：加载 → 拟合 OLS → 保存参数 → 保存诊断 → 打印摘要。"""
    logger = setup_logging()

    logger.info("Loading subsampled data")
    df = load_data(DATA_DIR / SUBSAMPLED_FILE)

    logger.info("Fitting OLS: %s ~ %s", TARGET, " + ".join(FEATURES))
    model = fit_ols(df)

    logger.info("Extracting parameter estimates")
    results_df = summarize_results(model)
    results_path = RESULTS_DIR / "frequentist_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved parameter results to %s", results_path)

    logger.info("Extracting model diagnostics")
    diag = extract_model_diagnostics(model)
    diag_df = pd.DataFrame([diag])
    diag_path = RESULTS_DIR / "frequentist_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    logger.info("Saved diagnostics to %s", diag_path)

    print_full_summary(model)


if __name__ == "__main__":
    main()
