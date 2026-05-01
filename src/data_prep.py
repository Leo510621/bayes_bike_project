"""
data_prep.py — 数据加载、验证与子采样

从 day.csv 加载原始数据，校验行数和完整性，
随机抽取 360 行并按日期排序，输出 day_subsampled.csv。
该文件是 B/C/D 模块的数据入口，列顺序遵循 config.COLUMN_ORDER 契约。
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    COLUMN_ORDER,
    DATA_DIR,
    RAW_DATA_FILE,
    SUBSAMPLE_N,
    SUBSAMPLED_FILE,
    RANDOM_SEED,
    setup_logging,
)


def load_raw_data(filepath: pd.io.common.PathLike) -> pd.DataFrame:
    """读取原始 CSV，解析 dteday 为日期类型。"""
    df = pd.read_csv(filepath, parse_dates=["dteday"])
    return df


def validate_data(df: pd.DataFrame) -> None:
    """校验原始数据：731 行、必要列存在、无缺失值。"""
    assert len(df) == 731, f"Expected 731 rows, got {len(df)}"
    required_cols = ["cnt", "temp", "hum", "dteday"]
    missing = set(required_cols) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df[required_cols].notna().all().all(), "Found NaN in required columns"


def subsample_data(
    df: pd.DataFrame,
    n: int = SUBSAMPLE_N,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """随机抽样 n 行，按日期升序排列，列顺序按契约重排。"""
    sampled = (
        df.sample(n=n, random_state=random_state)
        .sort_values("dteday")
        .reset_index(drop=True)
    )
    cols = [c for c in COLUMN_ORDER if c in sampled.columns]
    return sampled[cols]


def main() -> None:
    """执行完整数据准备流程：加载 → 验证 → 子采样 → 保存。"""
    logger = setup_logging()

    logger.info("Loading raw data from %s", DATA_DIR / RAW_DATA_FILE)
    df = load_raw_data(DATA_DIR / RAW_DATA_FILE)

    logger.info("Validating raw data")
    validate_data(df)

    logger.info("Subsampling to n=%d with random_state=%d", SUBSAMPLE_N, RANDOM_SEED)
    df_sub = subsample_data(df)

    output_path = DATA_DIR / SUBSAMPLED_FILE
    df_sub.to_csv(output_path, index=False)
    logger.info("Saved subsampled data to %s (%d rows)", output_path, len(df_sub))


if __name__ == "__main__":
    main()
