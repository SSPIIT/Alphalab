"""
value.py
────────
Computes value factors for all stocks in the OHLCV dataset.

WHAT IS THE VALUE FACTOR?
  The tendency of stocks trading cheaply relative to their fundamentals
  to outperform stocks trading expensively. First documented by Fama &
  French (1992) — one of the most studied factors in finance.

  The classic value metric is P/B (Price-to-Book ratio):
    P/B = Market Price per Share / Book Value per Share
    Low P/B = cheap (value stock), High P/B = expensive (growth stock)

  Other value proxies:
    P/E  = Price / Earnings per Share  (how much you pay per ₹1 of profit)
    P/S  = Price / Sales per Share     (how much you pay per ₹1 of revenue)
    EV/EBITDA = Enterprise Value / EBITDA (used for comparing across debt levels)

WHY USE PROXIES?
  yfinance gives us some fundamental data but not always complete.
  We use what's available and rank stocks relative to each other —
  the ranking matters more than the absolute number.

NOTE ON DATA AVAILABILITY:
  yfinance fundamental data for NSE stocks can be inconsistent.
  We handle missing data gracefully — stocks with no fundamental
  data get NaN scores and are excluded from that factor's ranking.

HOW THIS FITS IN THE PIPELINE:
  Raw OHLCV data (parquet) + yfinance fundamentals
      ↓
  value.py  ← you are here
      ↓
  data/features/value.parquet
"""

import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Fetches fundamental data for all tickers using yfinance.

    Returns a DataFrame with columns:
        [ticker, pe_ratio, pb_ratio, ps_ratio, market_cap]

    WHY FETCH SEPARATELY FROM OHLCV?
        Fundamentals change quarterly (earnings reports).
        Price data changes daily.
        Keeping them separate makes each easier to version and debug.
    """
    logger.info(f"Fetching fundamentals for {len(tickers)} stocks...")

    records = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            record = {
                "ticker": ticker,
                # trailingPE = P/E based on last 12 months earnings
                # forwardPE  = P/E based on next 12 months estimated earnings
                "pe_ratio": info.get("trailingPE", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "ps_ratio": info.get("priceToSalesTrailing12Months", np.nan),
                "market_cap": info.get("marketCap", np.nan),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda", np.nan),
            }
            records.append(record)
            logger.info(f"  {ticker}: P/E={record['pe_ratio']}, P/B={record['pb_ratio']}")

        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
            records.append({
                "ticker": ticker,
                "pe_ratio": np.nan,
                "pb_ratio": np.nan,
                "ps_ratio": np.nan,
                "market_cap": np.nan,
                "enterprise_to_ebitda": np.nan,
            })

    return pd.DataFrame(records)


def compute_value(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes value factor scores from fundamental data.

    VALUE SCORE LOGIC:
        Lower P/E, P/B, P/S = cheaper = higher value score.
        We invert the ratios so that high score = high value.

        Raw value = 1 / P/B  (higher = cheaper relative to book value)

        Then we rank cross-sectionally so scores are comparable
        to momentum and volatility ranks.

    WINSORIZATION:
        Extreme P/E ratios (e.g. 500x) distort rankings.
        We cap at the 5th and 95th percentile before ranking.
        This is standard practice in factor research.
    """
    df = fundamentals_df.copy()

    # ── Winsorize to remove extreme outliers ──────────────────────────────
    for col in ["pe_ratio", "pb_ratio", "ps_ratio"]:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower, upper=upper)

    # ── Value scores (inverted — lower ratio = higher value score) ────────
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    df["value_pe"] = 1 / (df["pe_ratio"] + eps)
    df["value_pb"] = 1 / (df["pb_ratio"] + eps)
    df["value_ps"] = 1 / (df["ps_ratio"] + eps)

    # ── Composite value score — equal weight average of available metrics ─
    value_cols = ["value_pe", "value_pb", "value_ps"]
    df["value_composite"] = df[value_cols].mean(axis=1)

    # ── Percentile ranks ──────────────────────────────────────────────────
    for col in value_cols + ["value_composite"]:
        df[f"{col}_rank"] = df[col].rank(pct=True, na_option="keep")

    logger.info(f"Value factors computed for {len(df)} stocks")
    logger.info(f"  value_composite stats: "
                f"mean={df['value_composite'].mean():.4f}, "
                f"std={df['value_composite'].std():.4f}")

    # Return only the columns we need downstream
    output_cols = [
        "ticker",
        "pe_ratio", "pb_ratio", "ps_ratio",
        "value_pe", "value_pb", "value_ps", "value_composite",
        "value_pe_rank", "value_pb_rank", "value_ps_rank", "value_composite_rank"
    ]

    return df[output_cols]


def run(raw_data_path: str, output_path: str) -> pd.DataFrame:
    """
    Main entry point — loads tickers, fetches fundamentals,
    computes value scores, saves output.
    """
    # Get ticker list from raw data
    df_raw = pd.read_parquet(raw_data_path)
    tickers = df_raw["ticker"].unique().tolist()

    # Fetch fundamentals from yfinance
    fundamentals = fetch_fundamentals(tickers)

    # Compute value scores
    value_df = compute_value(fundamentals)

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
        output_path="../../data/features/value.parquet"
    )

    print("\n── Top 10 Value Stocks (cheapest) ──")
    print(result.sort_values("value_composite_rank", ascending=False)
          .head(10)[["ticker", "pe_ratio", "pb_ratio", "value_composite_rank"]]
          .to_string(index=False))

    print("\n── Bottom 10 (most expensive) ──")
    print(result.sort_values("value_composite_rank", ascending=True)
          .head(10)[["ticker", "pe_ratio", "pb_ratio", "value_composite_rank"]]
          .to_string(index=False))