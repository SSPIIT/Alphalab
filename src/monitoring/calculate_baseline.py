"""
calculate_baseline.py
---------------------
Calculates drift baseline statistics for all factor columns in
data/features/all_factors.parquet and saves them to
data/features/baseline_stats.parquet

Baseline window: last 365 days of available data.

Statistics computed per column:
  - mean, std, min, max
  - percentiles: p5, p25, p50, p75, p95
  - null_rate: fraction of missing values
  - skewness, kurtosis

Run manually:
    python src/monitoring/calculate_baseline.py

Or called from Airflow for monthly baseline refresh.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # project root
FEATURES_PATH = ROOT / "data" / "features" / "all_factors.parquet"
BASELINE_PATH = ROOT / "data" / "features" / "baseline_stats.parquet"
BASELINE_META_PATH = ROOT / "data" / "features" / "baseline_meta.json"

# ── Config ──────────────────────────────────────────────────────────────────────
BASELINE_WINDOW_DAYS = 365          # rolling window for baseline
DATE_COL = "date"                   # name of the date column in all_factors
EXCLUDE_COLS = {DATE_COL, "ticker", "symbol", "open", "high", "low",
                "close", "volume", "adj_close"}   # non-factor columns to skip


# ── Helpers ─────────────────────────────────────────────────────────────────────
def load_factors(path: Path) -> pd.DataFrame:
    log.info(f"Loading factors from {path}")
    df = pd.read_parquet(path)

    # Normalise date column
    if DATE_COL not in df.columns:
        # Try index
        if df.index.name == DATE_COL or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": DATE_COL})
        else:
            raise ValueError(
                f"Could not find a '{DATE_COL}' column or DatetimeIndex in {path}"
            )

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    log.info(f"Loaded {len(df):,} rows, date range: "
             f"{df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    return df


def filter_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    cutoff = df[DATE_COL].max() - timedelta(days=days)
    filtered = df[df[DATE_COL] >= cutoff].copy()
    log.info(f"Baseline window (last {days}d): "
             f"{filtered[DATE_COL].min().date()} → {filtered[DATE_COL].max().date()}, "
             f"{len(filtered):,} rows")
    return filtered


def identify_factor_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    numeric = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    log.info(f"Found {len(numeric)} numeric factor columns: {numeric}")
    return numeric


def compute_baseline_stats(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame with one row per factor column containing all stats."""
    records = []

    for col in factor_cols:
        series = df[col].dropna()
        n_total = len(df[col])
        n_valid = len(series)

        if n_valid == 0:
            log.warning(f"Column '{col}' is entirely null — skipping.")
            continue

        arr = series.values.astype(float)

        record = {
            "feature":      col,
            "n_total":      n_total,
            "n_valid":      n_valid,
            "null_rate":    round(1 - n_valid / n_total, 6) if n_total > 0 else 1.0,
            "mean":         float(np.mean(arr)),
            "std":          float(np.std(arr, ddof=1)),
            "min":          float(np.min(arr)),
            "p5":           float(np.percentile(arr, 5)),
            "p25":          float(np.percentile(arr, 25)),
            "p50":          float(np.percentile(arr, 50)),
            "p75":          float(np.percentile(arr, 75)),
            "p95":          float(np.percentile(arr, 95)),
            "max":          float(np.max(arr)),
            "skewness":     float(stats.skew(arr)),
            "kurtosis":     float(stats.kurtosis(arr)),   # excess kurtosis
            "iqr":          float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }
        records.append(record)
        log.info(f"  {col:40s}  mean={record['mean']:+.4f}  std={record['std']:.4f}  "
                 f"null_rate={record['null_rate']:.2%}")

    baseline_df = pd.DataFrame(records).set_index("feature")
    return baseline_df


def save_baseline(baseline_df: pd.DataFrame,
                  raw_df: pd.DataFrame,
                  window_days: int) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save stats parquet
    baseline_df.to_parquet(BASELINE_PATH)
    log.info(f"Saved baseline stats → {BASELINE_PATH}  ({len(baseline_df)} features)")

    # Save lightweight JSON metadata alongside
    import json
    meta = {
        "created_at":        datetime.utcnow().isoformat() + "Z",
        "baseline_window_days": window_days,
        "data_start":        str(raw_df[DATE_COL].min().date()),
        "data_end":          str(raw_df[DATE_COL].max().date()),
        "n_rows":            len(raw_df),
        "n_features":        len(baseline_df),
        "features":          baseline_df.index.tolist(),
    }
    with open(BASELINE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved baseline metadata → {BASELINE_META_PATH}")


# ── Main ────────────────────────────────────────────────────────────────────────
def calculate_baseline(
    features_path: Path = FEATURES_PATH,
    baseline_path: Path = BASELINE_PATH,
    window_days: int = BASELINE_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Full pipeline: load → filter window → compute stats → save.
    Returns the baseline DataFrame so callers (Airflow, tests) can inspect it.
    """
    df = load_factors(features_path)
    windowed = filter_window(df, window_days)
    factor_cols = identify_factor_columns(windowed)

    if not factor_cols:
        raise RuntimeError("No numeric factor columns found — check EXCLUDE_COLS.")

    baseline_df = compute_baseline_stats(windowed, factor_cols)
    save_baseline(baseline_df, windowed, window_days)
    return baseline_df


if __name__ == "__main__":
    result = calculate_baseline()
    print("\n── Baseline Stats Preview ──")
    print(result[["mean", "std", "p25", "p50", "p75", "null_rate"]].to_string())