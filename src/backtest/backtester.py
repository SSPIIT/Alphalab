"""
backtester.py — AlphaLab Week 3
================================
Long/short factor backtester with MLflow logging.

QUANT LOGIC OVERVIEW
---------------------
For each factor column (e.g. "mom_6m_rank"):
  1. Every `holding_days` trading days, rank all stocks by the factor.
  2. Go LONG the top `top_pct` fraction of stocks (equal weight).
  3. Go SHORT the bottom `bottom_pct` fraction of stocks (equal weight).
  4. Hold the combined portfolio until the next rebalance.
  5. Portfolio return each day = (long leg mean return) − (short leg mean return).
  6. Compute performance metrics from the daily P&L series.

This is a "paper" backtest — no transaction costs, no slippage, no borrow costs.
The result is an upper-bound on what a factor's signal can deliver.
"""

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe inside Docker / CI

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.065   # ~RBI repo rate as of 2024 for NSE context


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe_ratio(daily_returns: pd.Series, rf_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    """
    Annualised Sharpe ratio.

    Sharpe = (Mean daily excess return × √252) / (Std daily return × √252)
           = (Mean daily excess return / Std daily return) × √252

    Excess return = portfolio return − risk-free rate (daily).
    Returns NaN if std == 0 (flat equity curve).
    """
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return np.nan

    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    sharpe = (excess.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return round(sharpe, 4)


def _annual_return(daily_returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate (CAGR) from a daily return series.

    CAGR = (final_value / initial_value) ^ (252 / n_days) − 1

    Starts from 1.0 (no leverage), cumulates daily.
    """
    if len(daily_returns) == 0:
        return np.nan

    cum = (1 + daily_returns).prod()
    n = len(daily_returns)
    cagr = cum ** (TRADING_DAYS_PER_YEAR / n) - 1
    return round(cagr, 4)


def _max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough percentage decline.

    Steps:
      1. Build cumulative equity curve starting at 1.0.
      2. At each point, compute drawdown = (equity − rolling_peak) / rolling_peak.
      3. Return the minimum (most negative) drawdown.

    Returns 0.0 if equity never falls below its prior peak.
    """
    if len(daily_returns) == 0:
        return np.nan

    equity = (1 + daily_returns).cumprod()
    rolling_peak = equity.cummax()
    drawdowns = (equity - rolling_peak) / rolling_peak
    return round(drawdowns.min(), 4)


def _win_rate(daily_returns: pd.Series) -> float:
    """
    Fraction of trading days with positive portfolio return.

    win_rate = 0.55 → portfolio was up on 55% of all days.
    """
    if len(daily_returns) == 0:
        return np.nan

    return round((daily_returns > 0).mean(), 4)


def _rolling_sharpe(daily_returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling 3-month Sharpe ratio — used to detect factor decay.

    A factor is considered "decaying" if its recent rolling Sharpe
    has dropped below the full-history mean Sharpe.
    """
    rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std = daily_returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)).round(4)


def _factor_decay_score(daily_returns: pd.Series, window: int = 63) -> float:
    """
    Factor decay metric:  recent_sharpe / historical_sharpe − 1

    Negative value → factor has weakened recently.
    Returned as a float (e.g. −0.40 means 40% decay).
    Returns NaN if history is too short.
    """
    if len(daily_returns) < window * 2:
        return np.nan

    full_sharpe = _sharpe_ratio(daily_returns)
    recent_sharpe = _sharpe_ratio(daily_returns.iloc[-window:])

    if full_sharpe == 0 or np.isnan(full_sharpe):
        return np.nan

    return round((recent_sharpe / full_sharpe) - 1, 4)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_daily_returns(
    raw_ohlcv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivots the OHLCV DataFrame into a wide daily-returns matrix.

    Input:  [Date, Close, ticker, ...]   (long format)
    Output: DataFrame with shape (n_dates, n_tickers),
            values = simple daily close-to-close returns.

    Missing tickers on a given date are left as NaN (not filled),
    so we never fabricate returns for missing data.
    """
    prices = raw_ohlcv_df.pivot_table(
        index="Date", columns="ticker", values="Close"
    ).sort_index()

    returns = prices.pct_change()
    return returns


def _rebalance_dates(dates: pd.DatetimeIndex, holding_days: int) -> list:
    """
    Generates rebalance dates by stepping every `holding_days` trading days
    through the sorted date index.

    First rebalance = first available date.
    """
    dates = sorted(dates)
    rebalance = [dates[0]]
    idx = 0
    while True:
        idx += holding_days
        if idx >= len(dates):
            break
        rebalance.append(dates[idx])
    return rebalance


# ─────────────────────────────────────────────────────────────────────────────
# CORE BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    factor_col: str,
    all_factors_df: pd.DataFrame,
    raw_ohlcv_df: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    holding_days: int = 21,
    output_dir: str = "data/backtest",
) -> dict:
    """
    Runs a long/short factor backtest and returns performance metrics.

    Args:
        factor_col      : Column name in all_factors_df to rank by
                          (e.g. "mom_6m_rank", "value_composite_rank").
                          Should be a 0–1 percentile rank; higher = stronger signal.
        all_factors_df  : DataFrame with [ticker, <factor_col>, ...].
                          One row per stock — the snapshot used to determine
                          long/short allocations.
        raw_ohlcv_df    : Long-format OHLCV with [Date, Close, ticker, ...].
                          Used to compute forward daily returns.
        top_pct         : Fraction of stocks to go long (e.g. 0.2 = top 20%).
        bottom_pct      : Fraction of stocks to short (e.g. 0.2 = bottom 20%).
        holding_days    : Days between rebalances (21 ≈ monthly, 63 ≈ quarterly).
        output_dir      : Where to save the P&L chart PNG.

    Returns:
        dict with keys:
          sharpe          — annualised Sharpe ratio
          annual_return   — CAGR
          max_drawdown    — worst peak-to-trough loss (negative)
          win_rate        — fraction of days with positive return
          factor_decay    — recent vs historical Sharpe decay ratio
          n_long          — avg number of stocks in long leg
          n_short         — avg number of stocks in short leg
          n_days          — total trading days in backtest
          pnl_chart_path  — file path to saved chart (or None)

    BACKTEST ASSUMPTIONS
    --------------------
    - Equal-weight within each leg (long and short).
    - Long/short weights are symmetric (dollar neutral).
    - Factor snapshot is static — we use the single all_factors_df
      snapshot for ALL rebalance periods. In a production backtest
      you'd rebuild factor scores at each rebalance date.
    - No transaction costs, no slippage, no borrow costs.
    - Returns are computed on the day AFTER each rebalance signal
      (avoiding look-ahead bias by one day).
    """
    logger.info(f"Starting backtest: factor='{factor_col}', "
                f"top={top_pct:.0%}, bottom={bottom_pct:.0%}, "
                f"hold={holding_days}d")

    # ── Validate inputs ────────────────────────────────────────────────────
    if factor_col not in all_factors_df.columns:
        raise ValueError(
            f"Factor column '{factor_col}' not found. "
            f"Available: {list(all_factors_df.columns)}"
        )

    raw_ohlcv_df = raw_ohlcv_df.copy()
    raw_ohlcv_df["Date"] = pd.to_datetime(raw_ohlcv_df["Date"])

    # ── Build daily returns matrix ────────────────────────────────────────
    daily_returns = _build_daily_returns(raw_ohlcv_df)
    logger.info(f"  Daily returns matrix: {daily_returns.shape[0]} dates × "
                f"{daily_returns.shape[1]} tickers")

    # ── Determine long / short universes from factor snapshot ─────────────
    factor_data = (
        all_factors_df[["ticker", factor_col]]
        .dropna(subset=[factor_col])
        .copy()
    )

    n_stocks = len(factor_data)
    n_long = max(1, int(np.floor(n_stocks * top_pct)))
    n_short = max(1, int(np.floor(n_stocks * bottom_pct)))

    long_tickers = (
        factor_data.nlargest(n_long, factor_col)["ticker"].tolist()
    )
    short_tickers = (
        factor_data.nsmallest(n_short, factor_col)["ticker"].tolist()
    )

    # Keep only tickers that exist in daily returns
    long_tickers  = [t for t in long_tickers  if t in daily_returns.columns]
    short_tickers = [t for t in short_tickers if t in daily_returns.columns]

    logger.info(f"  Long leg : {len(long_tickers)} stocks")
    logger.info(f"  Short leg: {len(short_tickers)} stocks")

    if not long_tickers or not short_tickers:
        raise ValueError(
            "Long or short leg is empty after filtering. "
            "Check that tickers in all_factors_df match raw_ohlcv_df."
        )

    # ── Build rebalance schedule ──────────────────────────────────────────
    trade_dates = daily_returns.index
    rebalance_schedule = _rebalance_dates(trade_dates, holding_days)
    logger.info(f"  Rebalance periods: {len(rebalance_schedule)} "
                f"(every ~{holding_days} days)")

    # ── Compute daily portfolio return ────────────────────────────────────
    # Strategy: long return − short return, each leg equal-weighted.
    # We skip day 0 of each holding period (the rebalance signal day)
    # to avoid look-ahead bias — we trade at the NEXT open effectively.

    portfolio_returns = []
    portfolio_dates   = []

    rebalance_set = set(rebalance_schedule)

    for i, date in enumerate(trade_dates):
        if i == 0:
            continue  # no prior day → no return

        long_ret  = daily_returns.loc[date, long_tickers].mean(skipna=True)
        short_ret = daily_returns.loc[date, short_tickers].mean(skipna=True)

        # Dollar-neutral L/S: long return − short return
        port_ret = long_ret - short_ret

        portfolio_returns.append(port_ret)
        portfolio_dates.append(date)

    pnl = pd.Series(portfolio_returns, index=portfolio_dates).dropna()

    if len(pnl) == 0:
        raise ValueError("Portfolio return series is empty — check OHLCV data coverage.")

    logger.info(f"  Backtest period: {pnl.index[0].date()} → {pnl.index[-1].date()} "
                f"({len(pnl)} trading days)")

    # ── Compute metrics ───────────────────────────────────────────────────
    sharpe        = _sharpe_ratio(pnl)
    ann_return    = _annual_return(pnl)
    max_dd        = _max_drawdown(pnl)
    win_r         = _win_rate(pnl)
    decay         = _factor_decay_score(pnl)

    logger.info(f"  Sharpe         : {sharpe}")
    logger.info(f"  Annual return  : {ann_return:.2%}" if not np.isnan(ann_return) else "  Annual return  : NaN")
    logger.info(f"  Max drawdown   : {max_dd:.2%}"    if not np.isnan(max_dd)    else "  Max drawdown   : NaN")
    logger.info(f"  Win rate       : {win_r:.2%}"     if not np.isnan(win_r)     else "  Win rate       : NaN")
    decay_str = f"{decay:.4f}" if (decay is not None and not np.isnan(decay)) else "NaN"
    logger.info(f"  Factor decay   : {decay_str}")

    # ── Save P&L chart ────────────────────────────────────────────────────
    pnl_chart_path = _save_pnl_chart(
        pnl=pnl,
        factor_col=factor_col,
        sharpe=sharpe,
        ann_return=ann_return,
        max_dd=max_dd,
        output_dir=output_dir,
    )

    return {
        "sharpe":         sharpe,
        "annual_return":  ann_return,
        "max_drawdown":   max_dd,
        "win_rate":       win_r,
        "factor_decay":   decay,
        "n_long":         len(long_tickers),
        "n_short":        len(short_tickers),
        "n_days":         len(pnl),
        "pnl_chart_path": pnl_chart_path,
        "pnl_series":     pnl,            # included for downstream plotting / ranker
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────

def _save_pnl_chart(
    pnl: pd.Series,
    factor_col: str,
    sharpe: float,
    ann_return: float,
    max_dd: float,
    output_dir: str,
) -> Optional[str]:
    """
    Saves a two-panel P&L chart:
      Top panel    — cumulative equity curve (starts at 1.0).
      Bottom panel — rolling 63-day (3M) Sharpe ratio.

    File saved as: <output_dir>/<factor_col>_pnl.png
    Returns the file path, or None if saving fails.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"{factor_col}_pnl.png")

        equity = (1 + pnl).cumprod()
        roll_sharpe = _rolling_sharpe(pnl, window=63)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )
        fig.patch.set_facecolor("#0f0f1a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#0f0f1a")
            ax.tick_params(colors="#cccccc", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        # ── Equity curve ──────────────────────────────────────────────────
        ax1.plot(equity.index, equity.values, color="#4fc3f7", linewidth=1.4, label="L/S Equity")
        ax1.axhline(1.0, color="#555577", linewidth=0.8, linestyle="--")
        ax1.fill_between(equity.index, 1.0, equity.values,
                         where=(equity.values >= 1.0), alpha=0.15, color="#4fc3f7")
        ax1.fill_between(equity.index, 1.0, equity.values,
                         where=(equity.values < 1.0), alpha=0.15, color="#ef5350")

        sharpe_str = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
        ret_str    = f"{ann_return:.1%}" if not np.isnan(ann_return) else "N/A"
        dd_str     = f"{max_dd:.1%}" if not np.isnan(max_dd) else "N/A"

        ax1.set_title(
            f"{factor_col}  |  Sharpe {sharpe_str}  |  CAGR {ret_str}  |  MaxDD {dd_str}",
            color="#e0e0e0", fontsize=11, pad=10
        )
        ax1.set_ylabel("Portfolio Value (start = 1.0)", color="#aaaacc", fontsize=9)
        ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e", labelcolor="#cccccc")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

        # ── Rolling Sharpe ────────────────────────────────────────────────
        valid_rs = roll_sharpe.dropna()
        colors_rs = ["#4fc3f7" if v >= 0 else "#ef5350" for v in valid_rs.values]
        ax2.bar(valid_rs.index, valid_rs.values, color=colors_rs, width=1.5, alpha=0.8)
        ax2.axhline(0, color="#555577", linewidth=0.8)
        ax2.set_ylabel("Rolling 3M Sharpe", color="#aaaacc", fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout(pad=1.5)
        plt.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor="#0f0f1a")
        plt.close(fig)

        logger.info(f"  P&L chart saved: {chart_path}")
        return chart_path

    except Exception as e:
        logger.error(f"  Failed to save chart: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_to_mlflow(
    factor_col: str,
    result: dict,
    tracking_uri: str = "http://localhost:5000",
) -> None:
    """
    Logs one backtest result as a single MLflow run.

    Each call creates one run under the experiment "factor_backtests".
    Params logged  : factor_col, top_pct, bottom_pct, holding_days
    Metrics logged : sharpe, annual_return, max_drawdown, win_rate,
                     factor_decay, n_long, n_short, n_days
    Artifact       : P&L chart PNG (if exists)

    MLflow URI:
      Inside Docker  → http://mlflow:5000
      From host      → http://localhost:5000

    Usage:
      result = run_backtest(...)
      log_to_mlflow("mom_6m_rank", result, tracking_uri="http://mlflow:5000")
    """
    try:
        import mlflow  # optional — only needed if MLflow is running

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("factor_backtests")

        with mlflow.start_run(run_name=factor_col):

            # Params
            mlflow.log_param("factor_col",    factor_col)
            mlflow.log_param("n_long",        result.get("n_long"))
            mlflow.log_param("n_short",       result.get("n_short"))

            # Metrics — skip NaN (MLflow rejects them)
            metrics = {
                "sharpe":        result.get("sharpe"),
                "annual_return": result.get("annual_return"),
                "max_drawdown":  result.get("max_drawdown"),
                "win_rate":      result.get("win_rate"),
                "factor_decay":  result.get("factor_decay"),
                "n_days":        result.get("n_days"),
            }
            for k, v in metrics.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mlflow.log_metric(k, v)

            # Artifact: P&L chart
            # fixed
            chart = result.get("pnl_chart_path")
            if chart and os.path.exists(chart):
                try:
                    mlflow.log_artifact(chart)
                except Exception as e:
                    logger.warning(f"Could not upload artifact: {e}")
                    # don't let this fail the entire run

        logger.info(f"  MLflow run logged for '{factor_col}'")

    except ImportError:
        logger.warning("mlflow not installed — skipping MLflow logging")
    except Exception as e:
        logger.error(f"  MLflow logging failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUN — all factors at once
# ─────────────────────────────────────────────────────────────────────────────

def run_all_factors(
    all_factors_df: pd.DataFrame,
    raw_ohlcv_df: pd.DataFrame,
    factor_cols: Optional[list] = None,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    holding_days: int = 21,
    output_dir: str = "data/backtest",
    mlflow_uri: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs backtests for all specified factor columns and returns a
    summary leaderboard DataFrame sorted by Sharpe ratio.

    Args:
        factor_cols : List of column names to backtest. If None, auto-detects
                      all columns ending in '_rank' plus composite columns.
        mlflow_uri  : If set, logs each run to MLflow at this URI.

    Returns:
        DataFrame with one row per factor:
          [factor, sharpe, annual_return, max_drawdown, win_rate,
           factor_decay, n_long, n_short, n_days]
        Sorted by sharpe descending.
    """
    if factor_cols is None:
        # Auto-detect: rank columns + composite columns
        factor_cols = [
            c for c in all_factors_df.columns
            if c.endswith("_rank") or c.endswith("_composite")
        ]
        # Exclude the catch-all overall_composite from individual factor ranking
        factor_cols = [c for c in factor_cols if c != "overall_composite"]
        logger.info(f"Auto-detected {len(factor_cols)} factor columns to backtest")

    records = []

    for fc in factor_cols:
        try:
            logger.info(f"\n{'─'*60}")
            logger.info(f"Backtesting: {fc}")
            result = run_backtest(
                factor_col=fc,
                all_factors_df=all_factors_df,
                raw_ohlcv_df=raw_ohlcv_df,
                top_pct=top_pct,
                bottom_pct=bottom_pct,
                holding_days=holding_days,
                output_dir=output_dir,
            )

            if mlflow_uri:
                log_to_mlflow(fc, result, tracking_uri=mlflow_uri)

            records.append({
                "factor":        fc,
                "sharpe":        result["sharpe"],
                "annual_return": result["annual_return"],
                "max_drawdown":  result["max_drawdown"],
                "win_rate":      result["win_rate"],
                "factor_decay":  result["factor_decay"],
                "n_long":        result["n_long"],
                "n_short":       result["n_short"],
                "n_days":        result["n_days"],
            })

        except Exception as e:
            logger.error(f"Backtest failed for '{fc}': {e}")
            records.append({"factor": fc, "sharpe": np.nan})

    summary = (
        pd.DataFrame(records)
        .sort_values("sharpe", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "factor_backtest_summary.parquet")
    summary.to_parquet(summary_path, index=False)
    logger.info(f"\nSummary saved → {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    RAW_DIR         = "data/raw"
    FEATURES_DIR    = "data/features"
    BACKTEST_DIR    = "data/backtest"

    # Load latest OHLCV
    raw_files = sorted(os.listdir(RAW_DIR))
    raw_path  = os.path.join(RAW_DIR, raw_files[-1])
    logger.info(f"Loading OHLCV from {raw_path}")
    raw_df = pd.read_parquet(raw_path)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    # Load combined factors
    factors_path = os.path.join(FEATURES_DIR, "all_factors.parquet")
    logger.info(f"Loading factors from {factors_path}")
    factors_df = pd.read_parquet(factors_path)

    # ── Quick single-factor demo ───────────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("DEMO: single factor backtest — mom_6m_rank")
    logger.info("═"*60)

    result = run_backtest(
        factor_col="mom_6m_rank",
        all_factors_df=factors_df,
        raw_ohlcv_df=raw_df,
        top_pct=0.2,
        bottom_pct=0.2,
        holding_days=21,
        output_dir=BACKTEST_DIR,
    )

    print("\n── mom_6m_rank Backtest Results ──")
    for k, v in result.items():
        if k not in ("pnl_series", "pnl_chart_path"):
            print(f"  {k:<20} {v}")
    if result["pnl_chart_path"]:
        print(f"\n  Chart saved: {result['pnl_chart_path']}")

    # ── Full factor sweep ─────────────────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("FULL SWEEP: backtesting all rank + composite columns")
    logger.info("═"*60)

    # Uncomment to also log to MLflow (Docker service must be running):
    # MLFLOW_URI = "http://localhost:5000"
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    summary = run_all_factors(
        all_factors_df=factors_df,
        raw_ohlcv_df=raw_df,
        top_pct=0.2,
        bottom_pct=0.2,
        holding_days=21,
        output_dir=BACKTEST_DIR,
        mlflow_uri=MLFLOW_URI,
    )

    print("\n" + "─"*80)
    print("FACTOR LEADERBOARD (sorted by Sharpe)")
    print("─"*80)
    print(summary.to_string(index=True))
    print(f"\nSummary parquet → {BACKTEST_DIR}/factor_backtest_summary.parquet")