"""
volatility.py
─────────────
Computes volatility factors for all stocks in the OHLCV dataset.

WHAT IS VOLATILITY (AS A FACTOR)?
  The tendency of LOW-volatility stocks to produce BETTER risk-adjusted
  returns than high-volatility stocks. Counterintuitive — but one of the
  most robust anomalies in finance (Baker, Bradley & Wurgler, 2011).

  This is called the "Low Volatility Anomaly" or "Defensive Factor".
  In a rational world, higher risk should mean higher return. In practice,
  overly volatile stocks are often overbought by speculators, causing them
  to underperform on a risk-adjusted basis.

  USE IN PORTFOLIO:
    - Low vol → defensive, capital-preservation tilt
    - High vol → aggressive, higher upside AND downside
    - Combined with momentum: low-vol + high-momentum = "quality momentum"

METRICS COMPUTED (all from OHLCV only):
  ┌──────────────────────┬─────────────────────────────────────────────────┐
  │ Factor               │ Description                                     │
  ├──────────────────────┼─────────────────────────────────────────────────┤
  │ hist_vol_1m          │ Annualised std of daily returns, 1-month window  │
  │ hist_vol_3m          │ Same, 3-month window                             │
  │ hist_vol_6m          │ Same, 6-month window (primary signal)            │
  │ atr_pct              │ ATR(14) as % of price — normalised for comparison│
  │ downside_vol_6m      │ Std of NEGATIVE returns only (Sortino denominator)│
  │ beta                 │ Sensitivity to index (market risk)               │
  │ max_drawdown_6m      │ Worst peak-to-trough loss in past 6 months       │
  └──────────────────────┴─────────────────────────────────────────────────┘

HOW THIS FITS IN THE PIPELINE:
  Raw OHLCV data (parquet)
      ↓
  volatility.py  ← you are here
      ↓
  data/features/volatility.parquet
      ↓
  combine_factors.py → all_factors.parquet (Day 5 final deliverable)
"""

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Trading days used for annualisation and window sizing
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
}
ATR_PERIOD = 14   # standard ATR lookback
MIN_HISTORY = 126  # minimum days needed (6M) to compute all metrics


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _daily_returns(close: pd.Series) -> pd.Series:
    """
    Computes simple daily returns from a closing price series.
    Returns NaN for the first row (no prior day).
    """
    return close.pct_change()


def _annualise_vol(daily_vol: float) -> float:
    """
    Converts daily return standard deviation to annualised volatility.
    Formula: σ_annual = σ_daily × √252
    """
    return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)


def _compute_atr(group: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """
    Computes the Average True Range (ATR) for the last `period` days.

    True Range = max(
        High - Low,
        |High - prev_Close|,
        |Low  - prev_Close|
    )
    ATR = rolling mean of True Range over `period` days.

    ATR captures intraday volatility (gaps + range), unlike return std
    which only captures close-to-close moves.
    """
    high = group["High"]
    low = group["Low"]
    close = group["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return atr


def _compute_beta(
    stock_returns: pd.Series,
    index_returns: pd.Series,
    window: int = TRADING_DAYS["6M"],
) -> float:
    """
    Computes Beta — the stock's sensitivity to index movements.

    Beta = Cov(stock, index) / Var(index)

    Beta > 1: stock amplifies index moves (aggressive)
    Beta < 1: stock dampens index moves (defensive)
    Beta ≈ 0: stock is uncorrelated to market

    Uses the overlapping window of dates between stock and index.
    Returns NaN if insufficient overlap.
    """
    # Align on common dates
    aligned = pd.concat([stock_returns, index_returns], axis=1, sort=False).dropna()
    aligned.columns = ["stock", "index"]

    if len(aligned) < window // 2:
        return np.nan

    # Use last `window` days
    aligned = aligned.iloc[-window:]

    cov_matrix = np.cov(aligned["stock"], aligned["index"])
    var_index = cov_matrix[1, 1]

    if var_index == 0 or np.isnan(var_index):
        return np.nan

    beta = cov_matrix[0, 1] / var_index
    return round(beta, 4)


# ──────────────────────────────────────────────
# CORE COMPUTATION
# ──────────────────────────────────────────────

def compute_volatility(
    df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Computes volatility factors for every stock.

    Args:
        df:       Raw OHLCV DataFrame with columns [Date, Open, High, Low, Close, ticker]
        index_df: Optional index OHLCV DataFrame (same format, single ticker = index)
                  Used for Beta computation. If None, Beta is set to NaN.

    Returns:
        DataFrame with columns:
          [ticker, hist_vol_1m, hist_vol_3m, hist_vol_6m,
           atr_pct, downside_vol_6m, beta, max_drawdown_6m]
        One row per stock — computed from the latest available window.

    NOTES ON EACH METRIC:
      hist_vol_*    — annualised, so 0.25 = 25% annual vol
      atr_pct       — ATR(14) divided by latest close price, so it's comparable
                      across stocks at different price levels
      downside_vol  — only uses negative return days; ignores upside swings
                      (the denominator of the Sortino ratio)
      beta          — requires index_df; skipped gracefully if not provided
      max_drawdown  — 0 means no drawdown; -0.30 means a 30% peak-to-trough loss
    """
    logger.info(f"Computing volatility factors for {df['ticker'].nunique()} stocks")

    # Prepare index returns for Beta (if provided)
    index_returns = None
    if index_df is not None and not index_df.empty:
        index_df = index_df.sort_values("Date").set_index("Date")
        index_returns = _daily_returns(index_df["Close"])
        logger.info("  Index returns prepared for Beta computation")
    else:
        logger.warning("  No index data — Beta will be NaN for all stocks")

    results = []

    for ticker, group in df.groupby("ticker"):
        try:
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) < MIN_HISTORY:
                logger.warning(
                    f"  {ticker}: only {len(group)} days of history "
                    f"(need {MIN_HISTORY}), skipping"
                )
                continue

            close = group["Close"]
            returns = _daily_returns(close).dropna()

            row = {"ticker": ticker}

            # ── Historical Volatility (1M, 3M, 6M) ────────────────────────
            # Annualised standard deviation of daily log returns
            for label, n_days in TRADING_DAYS.items():
                window_returns = returns.iloc[-n_days:]
                daily_std = window_returns.std()
                row[f"hist_vol_{label.lower()}"] = round(
                    _annualise_vol(daily_std), 6
                )

            # ── ATR % ──────────────────────────────────────────────────────
            # Requires High and Low columns
            if "High" in group.columns and "Low" in group.columns:
                atr = _compute_atr(group.tail(TRADING_DAYS["1M"] + ATR_PERIOD))
                latest_close = close.iloc[-1]
                row["atr_pct"] = round(atr / latest_close, 6) if latest_close > 0 else np.nan
            else:
                logger.warning(f"  {ticker}: High/Low columns missing — ATR skipped")
                row["atr_pct"] = np.nan

            # ── Downside Volatility (6M) ───────────────────────────────────
            # Same as hist_vol_6m but only counting negative return days
            # Captures "bad" volatility — used in Sortino ratio
            six_month_returns = returns.iloc[-TRADING_DAYS["6M"]:]
            negative_returns = six_month_returns[six_month_returns < 0]

            if len(negative_returns) > 5:  # need enough negative days
                row["downside_vol_6m"] = round(
                    _annualise_vol(negative_returns.std()), 6
                )
            else:
                row["downside_vol_6m"] = np.nan

            # ── Beta ───────────────────────────────────────────────────────
            if index_returns is not None:
                # Align stock returns to dates
                stock_returns_indexed = returns.copy()
                stock_returns_indexed.index = group["Date"].iloc[1:len(returns)+1].values
                stock_returns_series = pd.Series(
                    stock_returns_indexed.values,
                    index=pd.to_datetime(group["Date"].iloc[1:len(returns)+1].values)
                )
                row["beta"] = _compute_beta(stock_returns_series, index_returns)
            else:
                row["beta"] = np.nan

            # ── Maximum Drawdown (6M) ──────────────────────────────────────
            # Largest peak-to-trough percentage decline in the last 6 months
            # A measure of tail risk / worst-case loss
            six_month_close = close.iloc[-TRADING_DAYS["6M"]:]
            rolling_peak = six_month_close.cummax()
            drawdowns = (six_month_close - rolling_peak) / rolling_peak
            row["max_drawdown_6m"] = round(drawdowns.min(), 6)  # negative number

            results.append(row)

        except Exception as e:
            logger.error(f"  Failed volatility for {ticker}: {e}")
            continue

    if not results:
        raise ValueError("No volatility scores computed — check input data")

    result_df = pd.DataFrame(results)

    logger.info(f"Volatility computed for {len(result_df)} stocks")
    logger.info(f"  hist_vol_6m stats: mean={result_df['hist_vol_6m'].mean():.3f}, "
                f"std={result_df['hist_vol_6m'].std():.3f}")

    return result_df


# ──────────────────────────────────────────────
# RANKING
# ──────────────────────────────────────────────

def rank_volatility(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw volatility metrics into percentile ranks (0 to 1).

    RANKING DIRECTION — all inverted:
      Lower volatility → HIGHER rank (more attractive for defensive investing)

      hist_vol_*, atr_pct, downside_vol_6m, beta, max_drawdown_6m:
        all ranked DESCENDING so that rank=1.0 = lowest vol = most defensive

    WHY INVERT?
      We follow the Low Volatility Anomaly convention:
      the "best" stock from a vol factor perspective is the LEAST volatile.
      Consistent with momentum: rank=1.0 always means "most attractive".
    """
    ranked = vol_df.copy()

    vol_cols = [
        "hist_vol_1m", "hist_vol_3m", "hist_vol_6m",
        "atr_pct", "downside_vol_6m", "beta",
    ]
    for col in vol_cols:
        if col in ranked.columns:
            # ascending=False → lowest vol gets highest rank
            ranked[f"{col}_rank"] = ranked[col].rank(
                pct=True, ascending=False, na_option="keep"
            )

    # max_drawdown_6m is already negative (−0.3 is worse than −0.1)
    # ascending=False → least negative (smallest loss) gets highest rank
    if "max_drawdown_6m" in ranked.columns:
        ranked["max_drawdown_6m_rank"] = ranked["max_drawdown_6m"].rank(
            pct=True, ascending=False, na_option="keep"
        )

    logger.info("Volatility ranks computed")
    return ranked


def composite_volatility_score(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a single composite volatility score per stock by averaging
    all available rank columns.

    volatility_composite = 1.0 → lowest vol, most defensive stock
    volatility_composite = 0.0 → highest vol, most speculative stock

    The composite is itself percentile-ranked for consistency.
    """
    df = ranked_df.copy()

    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    if not rank_cols:
        logger.warning("No rank columns found — composite score skipped")
        df["volatility_composite"] = np.nan
        return df

    df["volatility_composite_raw"] = df[rank_cols].mean(axis=1, skipna=True)
    df["volatility_composite"] = df["volatility_composite_raw"].rank(
        pct=True, na_option="keep"
    )
    df.drop(columns=["volatility_composite_raw"], inplace=True)

    logger.info(f"  volatility_composite stats: "
                f"mean={df['volatility_composite'].mean():.3f}, "
                f"std={df['volatility_composite'].std():.3f}")
    return df


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def run(
    raw_data_path: str,
    output_path: str,
    index_data_path: str | None = None,
) -> pd.DataFrame:
    """
    Main entry point — loads data, computes volatility factors, saves output.

    Args:
        raw_data_path   : path to OHLCV parquet file (all stocks)
        output_path     : where to save volatility.parquet
        index_data_path : path to index OHLCV parquet (optional, for Beta)
                          The index file should have the same schema as raw data
                          but contain only the index (e.g. NIFTY 50)

    Returns:
        volatility DataFrame (also saved to output_path)
    """
    logger.info(f"Loading OHLCV data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    index_df = None
    if index_data_path and os.path.exists(index_data_path):
        logger.info(f"Loading index data from {index_data_path}")
        index_df = pd.read_parquet(index_data_path)
        index_df["Date"] = pd.to_datetime(index_df["Date"])
    else:
        logger.warning("No index data path provided — Beta will be NaN")

    # Compute raw volatility metrics
    vol_df = compute_volatility(df, index_df)

    # Add percentile ranks
    vol_df = rank_volatility(vol_df)

    # Add composite score
    vol_df = composite_volatility_score(vol_df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vol_df.to_parquet(output_path, index=False)
    logger.info(f"Volatility factors saved to {output_path}")

    return vol_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    raw_dir = "data/raw"
    files = sorted(os.listdir(raw_dir))
    latest = os.path.join(raw_dir, files[-1])

    result = run(
        raw_data_path=latest,
        output_path="data/features/volatility.parquet",
        index_data_path="data/raw/nifty50_index.parquet",
    )

    print("\n── Volatility Factor Sample (top 10 lowest vol) ──")
    print(result.sort_values("volatility_composite", ascending=False).head(10).to_string(index=False))
    print(f"\nShape: {result.shape}")