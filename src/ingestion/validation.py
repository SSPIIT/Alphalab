"""
validation.py
─────────────
Enhanced data validation for OHLCV parquet files.

WHY A SEPARATE FILE?
  Keeps the DAG file clean — it only handles orchestration.
  Validation logic lives here and can be unit tested independently.
  Week 2 Day 7 will write pytest tests specifically for these functions.
"""

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


def validate_ohlcv(file_path: str) -> dict:
    """
    Runs all validation checks on a parquet file.
    Returns a report dict — never raises, always logs.
    """
    report = {
        "file": file_path,
        "passed": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    # ── Check 1: File exists ───────────────────────────────────────────────
    if not os.path.exists(file_path):
        report["passed"] = False
        report["errors"].append(f"File not found: {file_path}")
        return report

    df = pd.read_parquet(file_path)

    # ── Check 2: Required columns ──────────────────────────────────────────
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "ticker"}
    missing = required - set(df.columns)
    if missing:
        report["passed"] = False
        report["errors"].append(f"Missing columns: {missing}")

    # ── Check 3: Not empty ─────────────────────────────────────────────────
    if df.empty:
        report["passed"] = False
        report["errors"].append("DataFrame is empty")
        return report

    # ── Check 4: Enough stocks ─────────────────────────────────────────────
    unique_tickers = df["ticker"].nunique()
    if unique_tickers < 80:
        report["passed"] = False
        report["errors"].append(f"Only {unique_tickers} stocks, expected 80+")

    # ── Check 5: Null close prices ─────────────────────────────────────────
    null_pct = df["Close"].isnull().mean() * 100
    if null_pct > 10:
        report["passed"] = False
        report["errors"].append(f"Null Close: {null_pct:.1f}%")
    elif null_pct > 2:
        report["warnings"].append(f"Elevated null Close: {null_pct:.1f}%")

    # ── Check 6: Negative prices ───────────────────────────────────────────
    neg = (df["Close"] < 0).sum()
    if neg > 0:
        report["passed"] = False
        report["errors"].append(f"{neg} negative Close prices")

    # ── Check 7: High >= Low sanity ───────────────────────────────────────
    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl > 0:
        report["warnings"].append(f"{bad_hl} rows where High < Low")

    # ── Check 8: Zero volume ──────────────────────────────────────────────
    zero_vol = (df["Volume"] == 0).sum()
    zero_vol_pct = zero_vol / len(df) * 100
    if zero_vol_pct > 5:
        report["warnings"].append(f"Zero volume rows: {zero_vol_pct:.1f}%")

    # ── Stats summary ──────────────────────────────────────────────────────
    report["stats"] = {
        "rows": len(df),
        "stocks": unique_tickers,
        "date_min": str(df["Date"].min()),
        "date_max": str(df["Date"].max()),
        "null_close_pct": round(null_pct, 2),
        "zero_volume_pct": round(zero_vol_pct, 2),
    }

    # ── Log the full report ────────────────────────────────────────────────
    if report["passed"]:
        logger.info(f"Validation PASSED for {file_path}")
    else:
        logger.error(f"Validation FAILED for {file_path}")

    for err in report["errors"]:
        logger.error(f"  ERROR: {err}")
    for warn in report["warnings"]:
        logger.warning(f"  WARNING: {warn}")

    logger.info(f"  Stats: {report['stats']}")

    return report