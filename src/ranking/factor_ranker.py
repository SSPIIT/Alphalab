"""
factor_ranker.py — AlphaLab Week 3
====================================
Loads backtest results (from MLflow or parquet fallback), ranks factors
by Sharpe ratio, and flags decaying factors.

FLOW
----
1. Try to load runs from MLflow experiment "factor_backtests".
2. If MLflow has no runs (or is unreachable), fall back to
   data/backtest/factor_backtest_summary.parquet produced by backtester.py.
3. Rank factors by Sharpe descending.
4. Flag each factor as "active" or "decaying" based on factor_decay score.
5. Save ranked output to data/backtest/factor_rankings.parquet.

DECAY DEFINITION
----------------
A factor is "decaying" if its factor_decay value < DECAY_THRESHOLD.
factor_decay = (recent_3M_sharpe / full_history_sharpe) - 1
  e.g. -0.40 means recent Sharpe is 40% worse than historical → decaying.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DECAY_THRESHOLD  = -0.30   # factor_decay below this → "decaying"
MIN_SHARPE_ACTIVE = 0.0    # factor must have positive Sharpe to be "active"
MLFLOW_EXPERIMENT = "factor_backtests"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FROM MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_mlflow(tracking_uri: str) -> pd.DataFrame | None:
    """
    Fetches all finished runs from the MLflow experiment "factor_backtests".

    Each run was created by backtester.log_to_mlflow() and contains:
      params  : factor_col, n_long, n_short
      metrics : sharpe, annual_return, max_drawdown, win_rate,
                factor_decay, n_days

    Returns:
        DataFrame with one row per factor run, or None if:
        - mlflow is not installed
        - experiment doesn't exist yet
        - experiment exists but has zero finished runs
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Check experiment exists
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if experiment is None:
            logger.warning(f"MLflow experiment '{MLFLOW_EXPERIMENT}' not found yet.")
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.sharpe DESC"],
        )

        if not runs:
            logger.warning("MLflow experiment exists but has no finished runs yet.")
            return None

        records = []
        for run in runs:
            row = {"factor": run.data.params.get("factor_col", run.info.run_name)}
            # Pull all logged metrics
            for metric in ["sharpe", "annual_return", "max_drawdown",
                           "win_rate", "factor_decay", "n_days"]:
                row[metric] = run.data.metrics.get(metric, np.nan)
            for param in ["n_long", "n_short"]:
                val = run.data.params.get(param)
                row[param] = int(val) if val is not None else np.nan
            records.append(row)

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} runs from MLflow ({tracking_uri})")
        return df

    except ImportError:
        logger.warning("mlflow not installed — skipping MLflow load")
        return None
    except Exception as e:
        logger.error(f"MLflow load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FROM PARQUET FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_parquet(summary_path: str) -> pd.DataFrame:
    """
    Loads the factor_backtest_summary.parquet written by backtester.run_all_factors().

    This is the fallback when MLflow has no runs yet.
    Raises FileNotFoundError if the parquet doesn't exist either
    (i.e. backtester hasn't been run yet at all).
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"No backtest results found.\n"
            f"  MLflow: no finished runs in '{MLFLOW_EXPERIMENT}'\n"
            f"  Parquet: {summary_path} does not exist\n"
            f"Run backtester.py first to generate results."
        )

    df = pd.read_parquet(summary_path)
    logger.info(f"Loaded {len(df)} factor results from parquet: {summary_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RANKING + STATUS
# ─────────────────────────────────────────────────────────────────────────────

def _assign_status(row: pd.Series) -> str:
    """
    Assigns a status label to each factor based on its metrics.

    Rules (in priority order):
      "decaying"  — factor_decay < DECAY_THRESHOLD
                    (recent performance significantly worse than historical)
      "weak"      — Sharpe <= 0 (factor has no positive risk-adjusted return)
      "active"    — everything else (healthy, positive Sharpe, not decaying)

    This status is used by:
      - FastAPI /factors endpoint (green/red badge in the frontend)
      - composite_model.py (to optionally exclude decaying factors)
    """
    sharpe = row.get("sharpe", np.nan)
    decay  = row.get("factor_decay", np.nan)

    if not np.isnan(decay) and decay < DECAY_THRESHOLD:
        return "decaying"
    if np.isnan(sharpe) or sharpe <= MIN_SHARPE_ACTIVE:
        return "weak"
    return "active"


def rank_factors(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks factors by Sharpe ratio and assigns status labels.

    Steps:
      1. Sort by Sharpe descending (NaN factors go to bottom).
      2. Add integer rank column (1 = best Sharpe).
      3. Add status column: "active" / "decaying" / "weak".
      4. Round floats for clean display.

    Args:
        raw_df: DataFrame from MLflow or parquet with columns:
                [factor, sharpe, annual_return, max_drawdown,
                 win_rate, factor_decay, n_long, n_short, n_days]

    Returns:
        Ranked DataFrame with added columns: [rank, status]
    """
    df = raw_df.copy()

    # Sort by Sharpe
    df = df.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # Assign status
    df["status"] = df.apply(_assign_status, axis=1)

    # Round display columns
    float_cols = ["sharpe", "annual_return", "max_drawdown", "win_rate", "factor_decay"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Log summary
    n_active   = (df["status"] == "active").sum()
    n_decaying = (df["status"] == "decaying").sum()
    n_weak     = (df["status"] == "weak").sum()

    logger.info(f"Factor ranking complete: {len(df)} factors total")
    logger.info(f"  active={n_active}  decaying={n_decaying}  weak={n_weak}")

    if n_active > 0:
        best = df[df["status"] == "active"].iloc[0]
        logger.info(f"  Best active factor: {best['factor']}  "
                    f"Sharpe={best['sharpe']:.3f}  "
                    f"CAGR={best['annual_return']:.2%}" if not np.isnan(best['annual_return'])
                    else f"  Best active factor: {best['factor']}  Sharpe={best['sharpe']:.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FILTERED VIEWS (used by composite_model + API)
# ─────────────────────────────────────────────────────────────────────────────

def get_active_factors(ranked_df: pd.DataFrame) -> list[str]:
    """
    Returns list of factor column names with status == "active",
    ordered by rank (best first).

    Used by composite_model.py to select which factors to feed into XGBoost.
    """
    active = ranked_df[ranked_df["status"] == "active"]["factor"].tolist()
    logger.info(f"Active factors ({len(active)}): {active}")
    return active


def get_decaying_factors(ranked_df: pd.DataFrame) -> list[str]:
    """
    Returns list of factor column names flagged as "decaying".
    Used for monitoring alerts and Grafana dashboards.
    """
    decaying = ranked_df[ranked_df["status"] == "decaying"]["factor"].tolist()
    if decaying:
        logger.warning(f"Decaying factors detected: {decaying}")
    return decaying


def get_top_n_factors(ranked_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Returns the top N ranked factors regardless of status.
    Used by the FastAPI /factors endpoint.
    """
    return ranked_df.head(n).copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    summary_parquet_path: str = "../../data/backtest/factor_backtest_summary.parquet",
    output_path: str = "../../data/backtest/factor_rankings.parquet",
    mlflow_uri: str | None = "http://localhost:5000",
) -> pd.DataFrame:
    """
    Full pipeline: load → rank → save → return.

    Args:
        summary_parquet_path : fallback parquet from backtester.run_all_factors()
        output_path          : where to save factor_rankings.parquet
        mlflow_uri           : MLflow tracking server URI.
                               Set to None to skip MLflow and always use parquet.

    Returns:
        Ranked DataFrame with columns:
          [rank, factor, sharpe, annual_return, max_drawdown,
           win_rate, factor_decay, n_long, n_short, n_days, status]
    """
    # ── Step 1: Load results ───────────────────────────────────────────────
    raw_df = None

    if mlflow_uri:
        logger.info(f"Attempting to load from MLflow: {mlflow_uri}")
        raw_df = _load_from_mlflow(mlflow_uri)

    if raw_df is None or raw_df.empty:
        logger.info("Falling back to parquet summary")
        raw_df = _load_from_parquet(summary_parquet_path)

    # ── Step 2: Rank + assign status ──────────────────────────────────────
    ranked_df = rank_factors(raw_df)

    # ── Step 3: Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ranked_df.to_parquet(output_path, index=False)
    logger.info(f"Factor rankings saved → {output_path}")

    return ranked_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    BACKTEST_DIR = "../../data/backtest"

    ranked = run(
        summary_parquet_path=os.path.join(BACKTEST_DIR, "factor_backtest_summary.parquet"),
        output_path=os.path.join(BACKTEST_DIR, "factor_rankings.parquet"),
        mlflow_uri="http://localhost:5000",
    )

    print("\n" + "─" * 75)
    print("FACTOR LEADERBOARD")
    print("─" * 75)

    display_cols = ["rank", "factor", "sharpe", "annual_return",
                    "max_drawdown", "win_rate", "factor_decay", "status"]
    available = [c for c in display_cols if c in ranked.columns]
    print(ranked[available].to_string(index=False))

    print("\n── Active factors (feed into composite_model.py) ──")
    print(get_active_factors(ranked))

    decaying = get_decaying_factors(ranked)
    if decaying:
        print(f"\n⚠️  Decaying factors: {decaying}")
    else:
        print("\n✓  No decaying factors detected")

    print(f"\nRankings saved → {BACKTEST_DIR}/factor_rankings.parquet")