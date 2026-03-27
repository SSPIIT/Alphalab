"""
momentum.py
───────────
Computes momentum factors for all stocks in the OHLCV dataset.

WHAT IS MOMENTUM?
  The tendency of stocks that have performed well recently to continue
  performing well in the near future. One of the most robust factors
  in academic finance — documented across 40+ years and 20+ markets.

  Formula: Momentum(N months) = (Close_today - Close_N_months_ago) / Close_N_months_ago

  We skip the most recent month (t-2 to t-13 for 12M momentum) because
  very short term momentum actually reverses — this is called the
  "1-month reversal" effect. Standard practice in quant finance.

HOW THIS FITS IN THE PIPELINE:
  Raw OHLCV data (parquet)
      ↓
  momentum.py  ← you are here
      ↓
  all_factors.parquet (one row per stock, one column per factor)
      ↓
  backtest.py (Week 3)
"""

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Trading days per month (NSE trades ~21 days/month)
TRADING_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes momentum factors for every stock.

    Args:
        df: Raw OHLCV dataframe with columns [Date, Close, ticker]

    Returns:
        DataFrame with columns [ticker, mom_1m, mom_3m, mom_6m, mom_12m]
        One row per stock — the LATEST momentum score for each.

    HOW IT WORKS:
        For each stock:
        1. Sort by date
        2. For each window (1M, 3M, 6M, 12M):
           - Find the close price N trading days ago
           - Compute return = (today - N days ago) / N days ago
        3. Collect the latest score into one row
    """
    logger.info(f"Computing momentum factors for {df['ticker'].nunique()} stocks")

    results = []

    for ticker, group in df.groupby("ticker"):
        try:
            # Sort by date — critical, returns are order-dependent
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) < TRADING_DAYS["12M"]:
                logger.warning(f"{ticker}: not enough history ({len(group)} days), skipping")
                continue

            latest_close = group["Close"].iloc[-1]

            row = {"ticker": ticker}

            for window_name, n_days in TRADING_DAYS.items():
                # Skip last month to avoid 1-month reversal effect
                # For 1M we don't skip (short term signal is still useful)
                if window_name == "1M":
                    past_close = group["Close"].iloc[-n_days]
                else:
                    # t-2 months to t-(N+1) months
                    skip = TRADING_DAYS["1M"]
                    past_close = group["Close"].iloc[-(n_days + skip)]

                if past_close == 0 or pd.isna(past_close):
                    row[f"mom_{window_name.lower()}"] = np.nan
                else:
                    momentum = (latest_close - past_close) / past_close
                    row[f"mom_{window_name.lower()}"] = round(momentum, 6)

            results.append(row)

        except Exception as e:
            logger.error(f"Failed momentum for {ticker}: {e}")
            continue

    if not results:
        raise ValueError("No momentum scores computed — check input data")

    result_df = pd.DataFrame(results)

    logger.info(f"Momentum computed for {len(result_df)} stocks")
    logger.info(f"  mom_6m stats: mean={result_df['mom_6m'].mean():.3f}, "
                f"std={result_df['mom_6m'].std():.3f}")

    return result_df


def rank_momentum(momentum_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw momentum returns into percentile ranks (0 to 1).

    WHY RANK INSTEAD OF RAW RETURN?
      Raw returns are hard to compare across factors.
      A momentum score of 0.15 means nothing on its own.
      A rank of 0.85 means "this stock is in the top 15% by momentum" —
      that's meaningful and comparable to value rank, volatility rank, etc.

      This is called cross-sectional ranking — ranking stocks against
      each other at the same point in time.
    """
    ranked = momentum_df.copy()

    for col in ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]:
        # pct=True gives percentile rank between 0 and 1
        # na_option="keep" leaves NaN as NaN instead of ranking it
        ranked[f"{col}_rank"] = ranked[col].rank(pct=True, na_option="keep")

    logger.info("Momentum ranks computed")
    return ranked


def run(raw_data_path: str, output_path: str) -> pd.DataFrame:
    """
    Main entry point — loads raw data, computes momentum, saves output.

    Args:
        raw_data_path : path to the OHLCV parquet file
        output_path   : where to save the momentum factor parquet

    Returns:
        momentum DataFrame (also saved to output_path)
    """
    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Compute raw momentum
    momentum_df = compute_momentum(df)

    # Add percentile ranks
    momentum_df = rank_momentum(momentum_df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    momentum_df.to_parquet(output_path, index=False)
    logger.info(f"Momentum factors saved to {output_path}")

    return momentum_df


if __name__ == "__main__":
    # Quick test — run directly to verify output
    # Usage: python momentum.py
    logging.basicConfig(level=logging.INFO)

    # Find the most recent parquet file in data/raw/
    raw_dir = "../../data/raw"
    files = sorted(os.listdir(raw_dir))
    latest = os.path.join(raw_dir, files[-1])

    result = run(
        raw_data_path=latest,
        output_path="../../data/features/momentum.parquet"
    )

    print("\n── Momentum Factor Sample ──")
    print(result.sort_values("mom_6m", ascending=False).head(10).to_string(index=False))
    print(f"\nShape: {result.shape}")