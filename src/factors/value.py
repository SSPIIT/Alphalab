"""
value.py
────────
Computes value factors for all stocks in the dataset.

WHAT IS VALUE?
  The tendency of "cheap" stocks (low price relative to fundamentals)
  to outperform "expensive" stocks over the long run. One of the oldest
  and most well-documented factors in finance — Fama & French (1992).

  Core idea: If a stock trades at a low multiple of its earnings, book
  value, or cash flows, it may be underpriced relative to intrinsic worth.

METRICS COMPUTED:
  ┌─────────────────┬──────────────────────────────────────────────────┐
  │ Factor          │ Formula                                          │
  ├─────────────────┼──────────────────────────────────────────────────┤
  │ P/E             │ Price / EPS  (lower = cheaper)                   │
  │ Earnings Yield  │ EPS / Price  (higher = cheaper, inverse of P/E)  │
  │ P/B             │ Price / Book Value per Share                     │
  │ Book-to-Market  │ Book Value / Market Cap  (higher = cheaper)      │
  │ EV/EBITDA       │ Enterprise Value / EBITDA                        │
  │ Dividend Yield  │ Annual Dividend per Share / Price                │
  └─────────────────┴──────────────────────────────────────────────────┘

  NOTE: Earnings Yield and Book-to-Market are the "investable" versions
  of P/E and P/B — they point in the same direction as momentum rank
  (higher rank = more attractive), making factor combination easier.

DATA INPUTS:
  This module accepts two DataFrames:
    1. ohlcv_df   — standard OHLCV with [Date, Close, ticker]
    2. fundamentals_df — one row per ticker with fundamental data
       Expected columns (all optional — missing ones are skipped):
         ticker, eps, book_value_per_share, ebitda, enterprise_value,
         dividend_per_share, market_cap, shares_outstanding

  If fundamentals_df is None or missing columns, only computable
  metrics are returned. The pipeline degrades gracefully.

HOW THIS FITS IN THE PIPELINE:
  Raw OHLCV data (parquet)  +  Fundamentals data (parquet, optional)
      ↓
  value.py  ← you are here
      ↓
  data/features/value.parquet
      ↓
  combine_factors.py → all_factors.parquet (Day 5 final deliverable)
"""

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divides two series, returning NaN wherever denominator is 0 or NaN.
    Prevents inf values from polluting downstream ranking.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (denominator == 0) | denominator.isna() | numerator.isna(),
            np.nan,
            numerator / denominator,
        )
    return pd.Series(result, index=numerator.index)


def _latest_price(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the most recent closing price per ticker from OHLCV data.

    Returns:
        DataFrame with columns [ticker, latest_close]
    """
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df["Date"] = pd.to_datetime(ohlcv_df["Date"])
    latest = (
        ohlcv_df.sort_values("Date")
        .groupby("ticker")["Close"]
        .last()
        .reset_index()
        .rename(columns={"Close": "latest_close"})
    )
    return latest


# ──────────────────────────────────────────────
# CORE COMPUTATION
# ──────────────────────────────────────────────

def compute_value(
    ohlcv_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Computes value factors for every stock.

    Args:
        ohlcv_df:        Raw OHLCV DataFrame [Date, Close, ticker, ...]
        fundamentals_df: Fundamentals DataFrame (optional). Expected columns:
                         ticker, eps, book_value_per_share, ebitda,
                         enterprise_value, dividend_per_share,
                         market_cap, shares_outstanding

    Returns:
        DataFrame with columns:
          [ticker, pe_ratio, earnings_yield, pb_ratio, book_to_market,
           ev_to_ebitda, dividend_yield]
        plus rank columns for each: [*_rank]
        One row per stock.

    RANKING CONVENTION:
        All rank columns are percentile ranks 0–1.
        Higher rank = more attractive from a VALUE perspective:
          - earnings_yield_rank, book_to_market_rank, dividend_yield_rank:
            higher raw value → higher rank  (cheaper = better)
          - pe_ratio_rank, pb_ratio_rank, ev_to_ebitda_rank:
            lower raw value → higher rank  (we invert these)
    """
    logger.info("Computing value factors")

    # Step 1: Get latest price per ticker
    prices = _latest_price(ohlcv_df)
    logger.info(f"  Prices loaded for {len(prices)} tickers")

    # Step 2: Merge with fundamentals if provided
    if fundamentals_df is not None and not fundamentals_df.empty:
        df = prices.merge(fundamentals_df, on="ticker", how="left")
        logger.info(f"  Fundamentals merged — {fundamentals_df.columns.tolist()}")
    else:
        logger.warning("  No fundamentals data provided — only price-based proxies computed")
        df = prices.copy()

    # ── P/E Ratio ──────────────────────────────────────────────────────────
    # Price / Earnings Per Share
    # Lower P/E → stock is "cheaper" relative to earnings
    # NaN if EPS is missing, zero, or negative (loss-making firms)
    if "eps" in df.columns:
        positive_eps = df["eps"].where(df["eps"] > 0)  # negative EPS → NaN
        df["pe_ratio"] = _safe_ratio(df["latest_close"], positive_eps).round(4)
        df["earnings_yield"] = _safe_ratio(positive_eps, df["latest_close"]).round(6)
    else:
        logger.warning("  'eps' column missing — pe_ratio and earnings_yield skipped")
        df["pe_ratio"] = np.nan
        df["earnings_yield"] = np.nan

    # ── P/B Ratio ──────────────────────────────────────────────────────────
    # Price / Book Value per Share
    # Lower P/B → stock trades closer to (or below) its accounting value
    # Book-to-Market is the Fama-French canonical form (higher = cheaper)
    if "book_value_per_share" in df.columns:
        positive_bv = df["book_value_per_share"].where(df["book_value_per_share"] > 0)
        df["pb_ratio"] = _safe_ratio(df["latest_close"], positive_bv).round(4)
        df["book_to_market"] = _safe_ratio(positive_bv, df["latest_close"]).round(6)
    elif "market_cap" in df.columns and "book_value_per_share" not in df.columns:
        logger.warning("  'book_value_per_share' missing — pb_ratio skipped")
        df["pb_ratio"] = np.nan
        df["book_to_market"] = np.nan
    else:
        df["pb_ratio"] = np.nan
        df["book_to_market"] = np.nan

    # ── EV/EBITDA ──────────────────────────────────────────────────────────
    # Enterprise Value / EBITDA
    # Preferred over P/E because it's capital-structure neutral
    # (accounts for debt, not just equity)
    if "enterprise_value" in df.columns and "ebitda" in df.columns:
        positive_ebitda = df["ebitda"].where(df["ebitda"] > 0)
        df["ev_to_ebitda"] = _safe_ratio(df["enterprise_value"], positive_ebitda).round(4)
    else:
        logger.warning("  'enterprise_value' or 'ebitda' missing — ev_to_ebitda skipped")
        df["ev_to_ebitda"] = np.nan

    # ── Dividend Yield ─────────────────────────────────────────────────────
    # Annual Dividend per Share / Price
    # Higher yield → more income relative to price (a value signal)
    # Many growth stocks pay no dividend → NaN is expected and normal
    if "dividend_per_share" in df.columns:
        df["dividend_yield"] = _safe_ratio(
            df["dividend_per_share"], df["latest_close"]
        ).round(6)
    else:
        logger.warning("  'dividend_per_share' missing — dividend_yield skipped")
        df["dividend_yield"] = np.nan

    # Select and return only factor columns
    factor_cols = [
        "ticker", "pe_ratio", "earnings_yield",
        "pb_ratio", "book_to_market",
        "ev_to_ebitda", "dividend_yield",
    ]
    result = df[factor_cols].copy()

    logger.info(f"Value factors computed for {len(result)} stocks")
    _log_coverage(result)

    return result


def _log_coverage(df: pd.DataFrame) -> None:
    """Logs what % of stocks have non-NaN values for each factor."""
    n = len(df)
    for col in df.columns:
        if col == "ticker":
            continue
        coverage = df[col].notna().sum()
        logger.info(f"  {col}: {coverage}/{n} stocks ({100*coverage/n:.0f}% coverage)")


# ──────────────────────────────────────────────
# RANKING
# ──────────────────────────────────────────────

def rank_value(value_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw value metrics into percentile ranks (0 to 1).

    RANKING DIRECTION:
      - earnings_yield, book_to_market, dividend_yield:
        rank ascending (higher raw = higher rank = more attractive)
      - pe_ratio, pb_ratio, ev_to_ebitda:
        rank descending (lower raw = higher rank = cheaper = more attractive)

    WHY INVERT SOME?
      We want rank=1.0 to always mean "most attractive value stock".
      A P/E of 5x is more attractive than P/E of 50x, so we flip
      the direction so that lower P/E → higher rank.
    """
    ranked = value_df.copy()

    # Higher is better → rank ascending
    for col in ["earnings_yield", "book_to_market", "dividend_yield"]:
        if col in ranked.columns:
            ranked[f"{col}_rank"] = ranked[col].rank(pct=True, na_option="keep")

    # Lower is better → rank descending (ascending=False)
    for col in ["pe_ratio", "pb_ratio", "ev_to_ebitda"]:
        if col in ranked.columns:
            ranked[f"{col}_rank"] = ranked[col].rank(
                pct=True, ascending=False, na_option="keep"
            )

    logger.info("Value ranks computed")
    return ranked


def composite_value_score(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a single composite value score per stock by averaging
    all available rank columns.

    WHY COMPOSITE?
      No single metric is perfect. P/E can be distorted by one-off
      earnings. P/B ignores intangibles. Averaging across metrics
      reduces noise and improves signal stability.

    The composite is itself percentile-ranked so it's on the same
    0–1 scale as all other factors.
    """
    df = ranked_df.copy()

    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    if not rank_cols:
        logger.warning("No rank columns found — composite score skipped")
        df["value_composite"] = np.nan
        return df

    # Row-wise mean across available ranks (ignores NaN automatically)
    df["value_composite_raw"] = df[rank_cols].mean(axis=1, skipna=True)

    # Re-rank the composite so it's comparable to other factor composites
    df["value_composite"] = df["value_composite_raw"].rank(pct=True, na_option="keep")
    df.drop(columns=["value_composite_raw"], inplace=True)

    logger.info(f"  value_composite stats: "
                f"mean={df['value_composite'].mean():.3f}, "
                f"std={df['value_composite'].std():.3f}")
    return df


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def run(
    raw_data_path: str,
    output_path: str,
    fundamentals_path: str | None = None,
) -> pd.DataFrame:
    """
    Main entry point — loads data, computes value factors, saves output.

    Args:
        raw_data_path     : path to OHLCV parquet file
        output_path       : where to save value.parquet
        fundamentals_path : path to fundamentals parquet (optional)
                            Expected columns: ticker, eps, book_value_per_share,
                            ebitda, enterprise_value, dividend_per_share

    Returns:
        value DataFrame (also saved to output_path)
    """
    logger.info(f"Loading OHLCV data from {raw_data_path}")
    ohlcv_df = pd.read_parquet(raw_data_path)
    ohlcv_df["Date"] = pd.to_datetime(ohlcv_df["Date"])

    fundamentals_df = None
    if fundamentals_path and os.path.exists(fundamentals_path):
        logger.info(f"Loading fundamentals from {fundamentals_path}")
        fundamentals_df = pd.read_parquet(fundamentals_path)
    else:
        logger.warning("No fundamentals file found — value factors will be NaN-heavy")

    # Compute raw value factors
    value_df = compute_value(ohlcv_df, fundamentals_df)

    # Add percentile ranks
    value_df = rank_value(value_df)

    # Add composite score
    value_df = composite_value_score(value_df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    value_df.to_parquet(output_path, index=False)
    logger.info(f"Value factors saved to {output_path}")

    return value_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    raw_dir = "../../data/raw"
    files = sorted(os.listdir(raw_dir))
    latest = os.path.join(raw_dir, files[-1])

    result = run(
        raw_data_path=latest,
        output_path="../../data/features/value.parquet",
        fundamentals_path="../../data/raw/fundamentals.parquet",
    )

    print("\n── Value Factor Sample (top 10 by composite) ──")
    print(result.sort_values("value_composite", ascending=False).head(10).to_string(index=False))
    print(f"\nShape: {result.shape}")