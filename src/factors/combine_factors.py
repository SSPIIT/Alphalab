"""
combine_factors.py
──────────────────
Merges all individual factor files into the final all_factors.parquet.

This is the Day 5 deliverable: /data/features/all_factors.parquet

PIPELINE POSITION:
  momentum.parquet    ─┐
  value.parquet       ─┤
  volatility.parquet  ─┼─→  combine_factors.py  →  all_factors.parquet
  mean_reversion.parquet ┘                              ↓
                                                   backtest.py (Week 3)

WHAT THIS SCRIPT DOES:
  1. Loads each individual factor parquet
  2. Merges them on `ticker` (outer join — keeps all stocks even if
     one factor file is missing some tickers)
  3. Computes an overall composite score across all factor composites
  4. Saves to all_factors.parquet

OUTPUT SCHEMA:
  One row per ticker. Columns include:

  From momentum.py:
    mom_1m, mom_3m, mom_6m, mom_12m
    mom_1m_rank, mom_3m_rank, mom_6m_rank, mom_12m_rank

  From value.py:
    pe_ratio, earnings_yield, pb_ratio, book_to_market, ev_to_ebitda, dividend_yield
    pe_ratio_rank, earnings_yield_rank, pb_ratio_rank, book_to_market_rank,
    ev_to_ebitda_rank, dividend_yield_rank, value_composite

  From volatility.py:
    hist_vol_1m, hist_vol_3m, hist_vol_6m, atr_pct, downside_vol_6m, beta, max_drawdown_6m
    hist_vol_1m_rank, ..., volatility_composite

  From mean_reversion.py:
    rsi_14, zscore_20d, zscore_60d, bb_pct_b, dist_from_52w_high,
    dist_from_52w_low, short_term_reversal
    rsi_14_rank, ..., mean_reversion_composite

  Combined:
    overall_composite  ← average of the four composite scores
"""

import logging
import os
import pandas as pd
import numpy as np
from functools import reduce

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

# Maps factor name → parquet file path
# Adjust paths to match your project structure
FACTOR_FILES = {
    "momentum":       "../../data/features/momentum.parquet",
    "value":          "../../data/features/value.parquet",
    "volatility":     "../../data/features/volatility.parquet",
    "mean_reversion": "../../data/features/mean_reversion.parquet",
}

OUTPUT_PATH = "../../data/features/all_factors.parquet"

# Composite columns from each factor module
# Used to compute overall_composite
COMPOSITE_COLS = {
    "momentum":       "mom_6m_rank",        # momentum has no explicit composite; use 6M rank
    "value":          "value_composite",
    "volatility":     "volatility_composite",
    "mean_reversion": "mean_reversion_composite",
}


# ──────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────

def load_factor_files(factor_files: dict[str, str]) -> dict[str, pd.DataFrame]:
    """
    Loads all available factor parquets. Skips files that don't exist
    with a warning (pipeline continues without them).

    Returns:
        Dict of {factor_name: DataFrame}
    """
    loaded = {}
    for name, path in factor_files.items():
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logger.info(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} cols")
            loaded[name] = df
        else:
            logger.warning(f"Factor file not found, skipping: {path}")
    return loaded


# ──────────────────────────────────────────────
# MERGE
# ──────────────────────────────────────────────

def merge_factors(factor_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-merges all factor DataFrames on `ticker`.

    Why outer join?
      Different factor modules may fail on different stocks (e.g. a stock
      with insufficient history for volatility but enough for momentum).
      Outer join preserves all tickers — NaN means "not computed", not
      "excluded from universe".

    Returns:
        Single DataFrame — one row per ticker, all factor columns.
    """
    if not factor_dfs:
        raise ValueError("No factor DataFrames to merge")

    dfs = list(factor_dfs.values())
    merged = reduce(
        lambda left, right: pd.merge(left, right, on="ticker", how="outer"),
        dfs
    )

    logger.info(f"Merged shape: {merged.shape} — {merged['ticker'].nunique()} unique tickers")
    return merged


# ──────────────────────────────────────────────
# OVERALL COMPOSITE
# ──────────────────────────────────────────────

def compute_overall_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a single overall composite score per stock by averaging
    all available factor composite columns.

    The weights here are equal. In a production system you'd tune these
    based on backtested IC (Information Coefficient) for each factor.

    overall_composite = mean(mom_6m_rank, value_composite,
                             volatility_composite, mean_reversion_composite)

    Then re-ranked to a 0–1 percentile for consistency.

    NOTE: momentum and mean_reversion are somewhat opposing signals —
    high momentum stocks often have low mean reversion scores. The
    composite naturally balances these out across different market regimes.
    """
    available_composites = [
        col for col in COMPOSITE_COLS.values() if col in df.columns
    ]

    if not available_composites:
        logger.warning("No composite columns found — overall_composite skipped")
        df["overall_composite"] = np.nan
        return df

    logger.info(f"Computing overall composite from: {available_composites}")

    df["overall_composite_raw"] = df[available_composites].mean(axis=1, skipna=True)
    df["overall_composite"] = df["overall_composite_raw"].rank(pct=True, na_option="keep")
    df.drop(columns=["overall_composite_raw"], inplace=True)

    logger.info(
        f"  overall_composite: mean={df['overall_composite'].mean():.3f}, "
        f"std={df['overall_composite'].std():.3f}"
    )
    return df


# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────

def log_summary(df: pd.DataFrame) -> None:
    """Logs coverage and basic stats for all factor columns."""
    n = len(df)
    logger.info(f"\n{'─'*60}")
    logger.info(f"all_factors.parquet summary — {n} stocks, {df.shape[1]} columns")
    logger.info(f"{'─'*60}")

    composite_cols = [c for c in df.columns if "composite" in c or c in COMPOSITE_COLS.values()]
    for col in composite_cols:
        if col in df.columns:
            coverage = df[col].notna().sum()
            mean = df[col].mean()
            logger.info(f"  {col:<35} {coverage}/{n} stocks, mean={mean:.3f}")

    logger.info(f"{'─'*60}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def run(
    factor_files: dict[str, str] | None = None,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Main entry point — loads all factor files, merges, computes composite,
    saves all_factors.parquet.

    Args:
        factor_files : dict mapping factor name → parquet path.
                       Defaults to FACTOR_FILES defined at top of file.
        output_path  : where to write all_factors.parquet

    Returns:
        all_factors DataFrame (also saved to output_path)
    """
    if factor_files is None:
        factor_files = FACTOR_FILES

    logger.info("Starting factor combination pipeline")

    # Load
    factor_dfs = load_factor_files(factor_files)

    if not factor_dfs:
        raise ValueError("No factor files could be loaded — run individual factor scripts first")

    # Merge
    all_factors = merge_factors(factor_dfs)

    # Overall composite
    all_factors = compute_overall_composite(all_factors)

    # Sort by overall composite descending (best stocks first)
    all_factors = all_factors.sort_values("overall_composite", ascending=False).reset_index(drop=True)

    # Summary
    log_summary(all_factors)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_factors.to_parquet(output_path, index=False)
    logger.info(f"\nall_factors.parquet saved to {output_path}")

    return all_factors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = run()

    print("\n── Top 10 Stocks by Overall Composite ──")
    display_cols = ["ticker", "overall_composite", "mom_6m_rank",
                    "value_composite", "volatility_composite", "mean_reversion_composite"]
    available = [c for c in display_cols if c in result.columns]
    print(result[available].head(10).to_string(index=False))

    print("\n── Bottom 10 Stocks by Overall Composite ──")
    print(result[available].tail(10).to_string(index=False))

    print(f"\nFinal shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")