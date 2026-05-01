"""
download_data.py — UCI Bike Sharing 数据集下载工具

从 UCI 官方仓库下载 Bike Sharing Dataset 的 zip 包，
解压出 day.csv 并校验行数、列名、缺失值。
支持主/备 URL 自动切换，已存在且合法时跳过下载。
"""

from __future__ import annotations

import logging
import shutil
import socket
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, setup_logging

# --- 下载源 ---
DATA_URL_PRIMARY = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)
DATA_URL_FALLBACK = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)
EXPECTED_ROWS = 731
EXPECTED_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]


def download_zip(url: str, dest_path: Path) -> Path:
    """从指定 URL 下载 zip 文件到 dest_path。"""
    logger = logging.getLogger("bayes_bike")
    logger.info("Downloading from %s", url)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DataDownloader/1.0)"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        total = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                logger.info("Downloaded %d KB", total // 1024)

    logger.info("Download complete: %s (%d KB)", dest_path.name, total // 1024)
    return dest_path


def extract_day_csv(zip_path: Path, output_dir: Path) -> Path:
    """从 zip 包中解压 day.csv 到 output_dir。"""
    logger = logging.getLogger("bayes_bike")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        day_entry = None
        for name in zf.namelist():
            if name.endswith("day.csv"):
                day_entry = name
                break

        if day_entry is None:
            raise FileNotFoundError("day.csv not found in zip archive")

        dest = output_dir / "day.csv"
        with zf.open(day_entry) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

    logger.info("Extracted day.csv (%d bytes)", dest.stat().st_size)
    return dest


def validate_download(csv_path: Path) -> None:
    """校验下载的 CSV：文件存在、731 行、16 列齐全、无缺失。"""
    assert csv_path.exists(), f"File not found: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    )

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    nan_cols = df[EXPECTED_COLUMNS].columns[df[EXPECTED_COLUMNS].isna().any()].tolist()
    assert not nan_cols, f"NaN values found in columns: {nan_cols}"


def download_with_fallback(
    primary_url: str, fallback_url: str, dest_path: Path
) -> Path:
    """尝试主 URL，失败后自动切换备用 URL。"""
    logger = logging.getLogger("bayes_bike")

    try:
        return download_zip(primary_url, dest_path)
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
        logger.warning("Primary URL failed: %s. Trying fallback...", e)

    try:
        return download_zip(fallback_url, dest_path)
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
        raise RuntimeError(
            f"Both download URLs failed. Last error: {e}\n"
            "Please manually download the zip from:\n"
            "  https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset\n"
            "Extract day.csv and place it in data/day.csv"
        ) from e


def main() -> None:
    """下载 day.csv（若已存在且合法则跳过）。"""
    logger = setup_logging()
    csv_path = DATA_DIR / "day.csv"

    if csv_path.exists():
        try:
            validate_download(csv_path)
            logger.info("Dataset already exists and is valid. Skipping download.")
            return
        except AssertionError as e:
            logger.warning("Existing file invalid (%s), re-downloading...", e)
            csv_path.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "bike_sharing.zip"

        download_with_fallback(DATA_URL_PRIMARY, DATA_URL_FALLBACK, zip_path)
        extracted = extract_day_csv(zip_path, tmp)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(csv_path))

    validate_download(csv_path)
    logger.info("Dataset ready: %s (%d rows)", csv_path, EXPECTED_ROWS)


if __name__ == "__main__":
    main()
