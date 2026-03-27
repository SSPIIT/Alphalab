"""
mean_reversion.py
─────────────────
Computes mean reversion factors for all stocks in the OHLCV dataset.

WHAT IS MEAN REVERSION?
  The tendency of stock prices to revert toward a long-run average after
  extreme moves. The counterpart to momentum — where momentum says
  "winners keep winning", mean reversion says "what goes too far, comes back".

  Both can be true at different timescales:
    - Short-term (days to weeks): mean reversion dominates
    - Medium-term (3–12 months): momentum dominates
    - Long-term (3–5 years): mean reversion dominates again

  For NSE stocks we focus on SHORT-TERM mean reversion signals
  (days to 1 month), which are orthogonal to the 12M momentum factor.

METRICS COMPUTED (all from OHLCV only):
  ┌──────────────────────────┬──────────────────────────────────────────────┐
  │ Factor                   │ Description                                  │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │ rsi_14                   │ RSI(14) — overbought/oversold oscillator     │
  │ zscore_20d               │ Z-score of price vs 20-day moving average    │
  │ zscore_60d               │ Z-score of price vs 60-day moving average    │
  │ bb_pct_b                 │ Bollinger Band %B — position within bands    │
  │ dist_from_52w_high       │ % below the 52-week high (reversal signal)   │
  │ dist_from_52w_low        │ % above the 52-week low (recovery signal)    │
  │ short_term_reversal      │ Negative of 1M return (raw reversal signal)  │
  └──────────────────────────┴──────────────────────────────────────────────┘

RANKING CONVENTION:
  rank=1.0 → most oversold / furthest below average → strongest mean
             reversion BUY candidate
  rank=0.0 → most overbought / furthest above average

HOW THIS FITS IN THE PIPELINE:
  Raw OHLCV data (parquet)
      ↓
  mean_reversion.py  ← you are here
      ↓
  data/features/mean_reversion.parquet
      ↓
  combine_factors.py → all_factors.parquet (Day 5 final deliverable)
"""

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MIN_HISTORY = 63       # 3 months minimum
RSI_PERIOD = 14
BB_PERIOD = 20         # Bollinger Band window
BB_STD_MULT = 2.0      # standard Bollinger Band width (±2 std)
ZSCORE_SHORT = 20      # days for short Z-score
ZSCORE_MED = 60        # days for medium Z-score
WEEKS_52 = 252         # trading days in a year (52w high/low)
REVERSAL_WINDOW = 21   # 1 month for short-term reversal


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float:
    """
    Computes the Relative Strength Index (RSI) for the latest date.

    RSI = 100 - (100 / (1 + RS))
    RS  = Average Gain / Average Loss over `period` days

    Interpretation:
      RSI > 70 → overbought (mean reversion: expect decline)
      RSI < 30 → oversold  (mean reversion: expect recovery)
      RSI = 50 → neutral

    We use Wilder's smoothing (exponential, as per original definition).
    """
    if len(close) < period + 1:
        return np.nan

    delta = close.diff().dropna()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Wilder's exponential moving average (alpha = 1/period)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0  # all gains, maximally overbought

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 4)


def _compute_zscore(close: pd.Series, window: int) -> float:
    """
    Computes the Z-score of the latest price relative to its rolling mean.

    Z = (price_today - mean(price, window)) / std(price, window)

    Z > 2  → price is 2 std above its average (overbought)
    Z < -2 → price is 2 std below its average (oversold)
    Z ≈ 0  → price is near its average (neutral)
    """
    if len(close) < window:
        return np.nan

    window_prices = close.iloc[-window:]
    mean = window_prices.mean()
    std = window_prices.std()

    if std == 0 or np.isnan(std):
        return np.nan

    return round((close.iloc[-1] - mean) / std, 4)


def _compute_bollinger_pct_b(close: pd.Series, period: int = BB_PERIOD, n_std: float = BB_STD_MULT) -> float:
    """
    Computes Bollinger Band %B for the latest date.

    %B = (Price - Lower Band) / (Upper Band - Lower Band)

    %B > 1.0 → price above upper band (overbought)
    %B < 0.0 → price below lower band (oversold)
    %B = 0.5 → price at the midline (neutral)

    %B is more interpretable than raw Z-score because it's bounded.
    """
    if len(close) < period:
        return np.nan

    window = close.iloc[-period:]
    sma = window.mean()
    std = window.std()

    if std == 0:
        return np.nan

    upper = sma + n_std * std
    lower = sma - n_std * std

    pct_b = (close.iloc[-1] - lower) / (upper - lower)
    return round(pct_b, 6)


def _compute_52w_distances(close: pd.Series) -> tuple[float, float]:
    """
    Computes how far the current price is from its 52-week high and low.

    dist_from_52w_high = (Price - 52w High) / 52w High  [negative or zero]
    dist_from_52w_low  = (Price - 52w Low) / 52w Low    [positive or zero]

    Large negative dist_from_52w_high → stock is far from its peak
    → potential mean reversion candidate (oversold relative to recent high)
    """
    if len(close) < WEEKS_52:
        window = close
    else:
        window = close.iloc[-WEEKS_52:]

    high_52w = window.max()
    low_52w = window.min()
    latest = close.iloc[-1]

    dist_high = round((latest - high_52w) / high_52w, 6) if high_52w > 0 else np.nan
    dist_low = round((latest - low_52w) / low_52w, 6) if low_52w > 0 else np.nan

    return dist_high, dist_low


# ──────────────────────────────────────────────
# CORE COMPUTATION
# ──────────────────────────────────────────────

def compute_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean reversion factors for every stock.

    Args:
        df: Raw OHLCV DataFrame with columns [Date, Close, ticker]
            (High and Low are used if available but not required)

    Returns:
        DataFrame with columns:
          [ticker, rsi_14, zscore_20d, zscore_60d, bb_pct_b,
           dist_from_52w_high, dist_from_52w_low, short_term_reversal]
        One row per stock — computed at the latest available date.
    """
    logger.info(f"Computing mean reversion factors for {df['ticker'].nunique()} stocks")

    results = []

    for ticker, group in df.groupby("ticker"):
        try:
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) < MIN_HISTORY:
                logger.warning(
                    f"  {ticker}: only {len(group)} days (need {MIN_HISTORY}), skipping"
                )
                continue

            close = group["Close"]
            row = {"ticker": ticker}

            # ── RSI ────────────────────────────────────────────────────────
            row["rsi_14"] = _compute_rsi(close, RSI_PERIOD)

            # ── Z-Score (20d and 60d) ──────────────────────────────────────
            row["zscore_20d"] = _compute_zscore(close, ZSCORE_SHORT)
            row["zscore_60d"] = _compute_zscore(close, ZSCORE_MED)

            # ── Bollinger %B ───────────────────────────────────────────────
            row["bb_pct_b"] = _compute_bollinger_pct_b(close, BB_PERIOD, BB_STD_MULT)

            # ── 52-Week High/Low Distance ──────────────────────────────────
            row["dist_from_52w_high"], row["dist_from_52w_low"] = _compute_52w_distances(close)

            # ── Short-Term Reversal (1M return, negated) ───────────────────
            # This is the direct counterpart to mom_1m in momentum.py
            # High recent return → negative reversal score (expect pullback)
            # Low recent return  → positive reversal score (expect bounce)
            if len(close) >= REVERSAL_WINDOW:
                one_month_return = (close.iloc[-1] - close.iloc[-REVERSAL_WINDOW]) / close.iloc[-REVERSAL_WINDOW]
                row["short_term_reversal"] = round(-one_month_return, 6)
            else:
                row["short_term_reversal"] = np.nan

            results.append(row)

        except Exception as e:
            logger.error(f"  Failed mean reversion for {ticker}: {e}")
            continue

    if not results:
        raise ValueError("No mean reversion scores computed — check input data")

    result_df = pd.DataFrame(results)

    logger.info(f"Mean reversion computed for {len(result_df)} stocks")
    logger.info(f"  rsi_14 stats: mean={result_df['rsi_14'].mean():.2f}, "
                f"std={result_df['rsi_14'].std():.2f}")
    logger.info(f"  zscore_20d stats: mean={result_df['zscore_20d'].mean():.3f}, "
                f"std={result_df['zscore_20d'].std():.3f}")

    return result_df


# ──────────────────────────────────────────────
# RANKING
# ──────────────────────────────────────────────

def rank_mean_reversion(mr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw mean reversion metrics into percentile ranks (0 to 1).

    RANKING DIRECTION — all oriented so rank=1.0 = strongest BUY signal:

      RSI: ascending=False → lowest RSI (most oversold) gets rank=1.0
      zscore_*: ascending=False → most negative Z-score gets rank=1.0
      bb_pct_b: ascending=False → lowest %B (below lower band) gets rank=1.0
      dist_from_52w_high: ascending=True → most negative (furthest from peak) gets rank=1.0
      dist_from_52w_low: ascending=False → lowest ratio (closest to low) gets rank=1.0
      short_term_reversal: ascending=False → most negative 1M return gets rank=1.0
    """
    ranked = mr_df.copy()

    # Lower raw value → stronger reversion signal → higher rank
    descending_cols = [
        "rsi_14", "zscore_20d", "zscore_60d",
        "bb_pct_b", "short_term_reversal",
    ]
    for col in descending_cols:
        if col in ranked.columns:
            ranked[f"{col}_rank"] = ranked[col].rank(
                pct=True, ascending=False, na_option="keep"
            )

    # dist_from_52w_high is negative: most negative → furthest from peak → rank=1.0
    # ascending=True so -0.50 (further) > -0.05 (closer) in rank
    if "dist_from_52w_high" in ranked.columns:
        ranked["dist_from_52w_high_rank"] = ranked["dist_from_52w_high"].rank(
            pct=True, ascending=True, na_option="keep"
        )

    # dist_from_52w_low is positive: lower → closer to 52w low → oversold → rank=1.0
    if "dist_from_52w_low" in ranked.columns:
        ranked["dist_from_52w_low_rank"] = ranked["dist_from_52w_low"].rank(
            pct=True, ascending=False, na_option="keep"
        )

    logger.info("Mean reversion ranks computed")
    return ranked


def composite_mean_reversion_score(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a single composite mean reversion score per stock.

    mean_reversion_composite = 1.0 → most oversold, strongest bounce candidate
    mean_reversion_composite = 0.0 → most overbought, risk of pullback

    Note: By construction, mean_reversion_composite and momentum's
    mom_1m_rank will tend to be negatively correlated — that's expected.
    In a multi-factor model you'd weight these carefully or use only
    one timescale's signal.
    """
    df = ranked_df.copy()

    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    if not rank_cols:
        logger.warning("No rank columns found — composite skipped")
        df["mean_reversion_composite"] = np.nan
        return df

    df["mean_reversion_composite_raw"] = df[rank_cols].mean(axis=1, skipna=True)
    df["mean_reversion_composite"] = df["mean_reversion_composite_raw"].rank(
        pct=True, na_option="keep"
    )
    df.drop(columns=["mean_reversion_composite_raw"], inplace=True)

    logger.info(f"  mean_reversion_composite stats: "
                f"mean={df['mean_reversion_composite'].mean():.3f}, "
                f"std={df['mean_reversion_composite'].std():.3f}")
    return df


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def run(raw_data_path: str, output_path: str) -> pd.DataFrame:
    """
    Main entry point — loads raw OHLCV, computes mean reversion, saves output.

    Args:
        raw_data_path : path to the OHLCV parquet file
        output_path   : where to save mean_reversion.parquet

    Returns:
        mean reversion DataFrame (also saved to output_path)
    """
    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Compute raw metrics
    mr_df = compute_mean_reversion(df)

    # Add percentile ranks
    mr_df = rank_mean_reversion(mr_df)

    # Add composite score
    mr_df = composite_mean_reversion_score(mr_df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mr_df.to_parquet(output_path, index=False)
    logger.info(f"Mean reversion factors saved to {output_path}")

    return mr_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    raw_dir = "../../data/raw"
    files = sorted(os.listdir(raw_dir))
    latest = os.path.join(raw_dir, files[-1])

    result = run(
        raw_data_path=latest,
        output_path="../../data/features/mean_reversion.parquet",
    )

    print("\n── Mean Reversion Factor Sample (top 10 oversold) ──")
    print(result.sort_values("mean_reversion_composite", ascending=False).head(10).to_string(index=False))
    print(f"\nShape: {result.shape}")