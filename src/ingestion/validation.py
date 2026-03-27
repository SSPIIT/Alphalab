
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

    if not os.path.exists(file_path):
        report["passed"] = False
        report["errors"].append(f"File not found: {file_path}")
        return report

    df = pd.read_parquet(file_path)

    required = {"Date", "Open", "High", "Low", "Close", "Volume", "ticker"}
    missing = required - set(df.columns)
    if missing:
        report["passed"] = False
        report["errors"].append(f"Missing columns: {missing}")

    if df.empty:
        report["passed"] = False
        report["errors"].append("DataFrame is empty")
        return report

    unique_tickers = df["ticker"].nunique()
    if unique_tickers < 80:
        report["passed"] = False
        report["errors"].append(f"Only {unique_tickers} stocks, expected 80+")

    null_pct = df["Close"].isnull().mean() * 100
    if null_pct > 10:
        report["passed"] = False
        report["errors"].append(f"Null Close: {null_pct:.1f}%")
    elif null_pct > 2:
        report["warnings"].append(f"Elevated null Close: {null_pct:.1f}%")

    neg = (df["Close"] < 0).sum()
    if neg > 0:
        report["passed"] = False
        report["errors"].append(f"{neg} negative Close prices")

    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl > 0:
        report["warnings"].append(f"{bad_hl} rows where High < Low")

    zero_vol = (df["Volume"] == 0).sum()
    zero_vol_pct = zero_vol / len(df) * 100
    if zero_vol_pct > 5:
        report["warnings"].append(f"Zero volume rows: {zero_vol_pct:.1f}%")

    report["stats"] = {
        "rows": len(df),
        "stocks": unique_tickers,
        "date_min": str(df["Date"].min()),
        "date_max": str(df["Date"].max()),
        "null_close_pct": round(null_pct, 2),
        "zero_volume_pct": round(zero_vol_pct, 2),
    }

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