"""
shap_explainer.py — AlphaLab Week 3
======================================
Loads the trained XGBoost composite model and computes SHAP values
to explain per-stock alpha score predictions.

WHAT IS SHAP?
-------------
SHAP (SHapley Additive exPlanations) answers:
  "Why did this stock get a HIGH / LOW alpha score?"

For each stock, SHAP decomposes the model's prediction into contributions
from each factor:

  predicted_return = base_value
                   + shap(mom_12m_rank)
                   + shap(mom_6m_rank)
                   + shap(dist_from_52w_high_rank)
                   + shap(mom_3m_rank)

A positive SHAP value means the factor PUSHED the score UP.
A negative SHAP value means the factor PULLED the score DOWN.

OUTPUT PER STOCK
----------------
  top_factor_1, top_factor_2, top_factor_3  — names of the 3 biggest drivers
  shap_1, shap_2, shap_3                    — their SHAP contribution values
  explanation                               — human-readable sentence

USAGE
-----
  Run composite_model.py first to generate:
    data/models/composite_model.json
    data/models/composite_scores.parquet

  Then run:
    python3 shap_explainer.py
"""

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Active factors — must match composite_model.py
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

ACTIVE_FACTORS = params["model"]["active_factors"]

# Human-readable factor labels for explanation sentences
FACTOR_LABELS = {
    "mom_12m_rank":          "12-month momentum",
    "mom_6m_rank":           "6-month momentum",
    "mom_3m_rank":           "3-month momentum",
    "mom_1m_rank":           "1-month momentum",
    "dist_from_52w_high_rank": "distance from 52-week high",
    "value_composite_rank":  "value composite",
    "volatility_composite":  "low volatility",
    "mean_reversion_composite": "mean reversion",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Loads the saved XGBoost model from a .json file.
    Raises FileNotFoundError with a clear message if not found.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Run composite_model.py first to generate it."
        )

    model = XGBRegressor()
    model.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def load_feature_matrix(
    all_factors_path: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Loads the latest factor snapshot and extracts the feature matrix.

    Fills NaN with column median — same logic as composite_model.score_stocks()
    so SHAP values are consistent with the scores already produced.

    Returns:
        df_clean : DataFrame with [ticker] + feature_cols, no NaN in features
    """
    df = pd.read_parquet(all_factors_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from all_factors.parquet: {missing}")

    df = df[["ticker"] + feature_cols].copy()

    for col in feature_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.warning(f"  {col}: filled {n_nan} NaN with median {median:.4f}")

    logger.info(f"Feature matrix: {len(df)} stocks × {len(feature_cols)} features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE SHAP VALUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Computes SHAP values using TreeExplainer (fast, exact for XGBoost).

    TreeExplainer uses the tree structure directly — no sampling needed.
    Returns a matrix of shape (n_stocks, n_features).

    shap_values[i, j] = contribution of feature j to stock i's prediction.
    Sum of shap_values[i, :] + expected_value ≈ model.predict(X)[i]
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: pip install shap")

    logger.info(f"Computing SHAP values for {len(X)} stocks...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    logger.info(f"  SHAP matrix shape: {shap_values.shape}")
    logger.info(f"  Expected value (base rate): {explainer.expected_value:.6f}")

    return shap_values, explainer.expected_value


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PER-STOCK EXPLANATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_explanation_table(
    tickers: list[str],
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_cols: list[str],
    scores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Builds a per-stock explanation table with top-3 SHAP drivers.

    For each stock:
      - Ranks features by |SHAP value| (absolute contribution)
      - Records top 3 feature names and their signed SHAP values
      - Generates a plain-English explanation sentence

    Args:
        tickers      : list of ticker strings (same order as X)
        X            : feature matrix used for SHAP
        shap_values  : SHAP values array (n_stocks × n_features)
        feature_cols : ordered list of feature column names
        scores_df    : optional — merged in for predicted_return and
                       composite_alpha_score columns

    Returns:
        DataFrame with one row per stock, sorted by |total SHAP| descending.
    """
    records = []

    for i, ticker in enumerate(tickers):
        sv   = shap_values[i]           # shape: (n_features,)
        absv = np.abs(sv)

        # Top 3 by absolute SHAP value
        top3_idx = np.argsort(absv)[::-1][:3]

        row = {"ticker": ticker}

        for rank, idx in enumerate(top3_idx, start=1):
            fname       = feature_cols[idx]
            shap_val    = round(sv[idx], 6)
            fval        = round(X.iloc[i][fname], 4)

            row[f"top_factor_{rank}"] = fname
            row[f"shap_{rank}"]       = shap_val
            row[f"factor_value_{rank}"] = fval

        # Total SHAP magnitude (how "explainable" this stock's score is)
        row["total_shap_magnitude"] = round(absv.sum(), 6)

        # Human-readable explanation
        row["explanation"] = _build_explanation(ticker, top3_idx, sv, feature_cols, X.iloc[i])

        records.append(row)

    explanation_df = pd.DataFrame(records)

    # Merge with scores if provided
    if scores_df is not None:
        merge_cols = ["ticker", "predicted_return", "composite_alpha_score",
                      "alpha_rank", "recommendation"]
        available  = [c for c in merge_cols if c in scores_df.columns]
        explanation_df = explanation_df.merge(
            scores_df[available], on="ticker", how="left"
        )
        explanation_df = explanation_df.sort_values(
            "composite_alpha_score", ascending=False
        ).reset_index(drop=True)

    logger.info(f"Explanation table built: {len(explanation_df)} stocks")
    return explanation_df


def _build_explanation(
    ticker: str,
    top3_idx: np.ndarray,
    shap_vals: np.ndarray,
    feature_cols: list[str],
    feature_row: pd.Series,
) -> str:
    """
    Generates a plain-English sentence explaining a stock's alpha score.

    Example output:
      "ONGC.NS scored high mainly due to strong 12-month momentum
       (SHAP +0.023), supported by strong 6-month momentum (SHAP +0.018),
       partially offset by weak distance from 52-week high (SHAP -0.004)."
    """
    parts = []

    for rank, idx in enumerate(top3_idx):
        fname    = feature_cols[idx]
        sv       = shap_vals[idx]
        fval     = feature_row[fname]
        label    = FACTOR_LABELS.get(fname, fname.replace("_", " "))

        direction = "strong" if fval > 0.5 else "weak"
        impact    = "boosted" if sv > 0 else "dragged down"

        part = f"{direction} {label} (SHAP {sv:+.4f})"

        if rank == 0:
            parts.append(f"{ticker} score mainly {impact} by {part}")
        elif rank == 1:
            parts.append(f"supported by {part}" if sv > 0 else f"offset by {part}")
        else:
            parts.append(f"and {part}")

    return ", ".join(parts) + "."


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SUMMARY — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def global_shap_summary(
    shap_values: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Computes global feature importance as mean |SHAP| across all stocks.

    This is the SHAP-based alternative to XGBoost's built-in feature_importances_
    — it's more interpretable because it's in the same units as the prediction.

    mean_abs_shap = average absolute contribution to predicted return.
    e.g. 0.015 means this factor moves predicted return by ~1.5% on average.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    summary = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": mean_abs_shap.round(6),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    summary["rank"] = summary.index + 1

    logger.info("Global SHAP summary:")
    for _, row in summary.iterrows():
        logger.info(f"  {row['rank']}. {row['feature']:<30} mean|SHAP|={row['mean_abs_shap']:.6f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    model_path: str          = "../../data/models/composite_model.json",
    all_factors_path: str    = "../../data/features/all_factors.parquet",
    scores_path: str         = "../../data/models/composite_scores.parquet",
    output_dir: str          = "data/models",
    feature_cols: list[str]  = None,
) -> pd.DataFrame:
    """
    Full SHAP pipeline: load → compute → explain → save.

    Args:
        model_path       : path to composite_model.json
        all_factors_path : path to all_factors.parquet (feature snapshot)
        scores_path      : path to composite_scores.parquet (for merging scores)
        output_dir       : where to save shap_explanations.parquet
        feature_cols     : feature columns to explain.
                           Defaults to ACTIVE_FACTORS if None.

    Returns:
        explanation_df : per-stock explanation table
    """
    if feature_cols is None:
        feature_cols = ACTIVE_FACTORS

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(model_path)

    # ── Load feature matrix ───────────────────────────────────────────────
    df_features = load_feature_matrix(all_factors_path, feature_cols)
    tickers     = df_features["ticker"].tolist()
    X           = df_features[feature_cols].reset_index(drop=True)

    # ── Load scores (optional, for merging) ──────────────────────────────
    scores_df = None
    if os.path.exists(scores_path):
        scores_df = pd.read_parquet(scores_path)
        logger.info(f"Loaded composite scores: {len(scores_df)} stocks")
    else:
        logger.warning(f"Scores not found at {scores_path} — skipping merge")

    # ── Compute SHAP values ───────────────────────────────────────────────
    shap_values, base_value = compute_shap_values(model, X)

    # ── Global summary ────────────────────────────────────────────────────
    global_summary = global_shap_summary(shap_values, feature_cols)

    # ── Per-stock explanations ────────────────────────────────────────────
    explanation_df = build_explanation_table(
        tickers=tickers,
        X=X,
        shap_values=shap_values,
        feature_cols=feature_cols,
        scores_df=scores_df,
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    explanations_path = os.path.join(output_dir, "shap_explanations.parquet")
    explanation_df.to_parquet(explanations_path, index=False)
    logger.info(f"SHAP explanations saved → {explanations_path}")

    global_path = os.path.join(output_dir, "shap_global_summary.parquet")
    global_summary.to_parquet(global_path, index=False)
    logger.info(f"Global SHAP summary saved → {global_path}")

    # Save base value for API use
    with open(os.path.join(output_dir, "shap_base_value.json"), "w") as f:
        json.dump({"base_value": float(base_value)}, f)

    return explanation_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    MODEL_PATH   = os.path.join(BASE_DIR, "data/models/composite_model.json")
    FACTORS_PATH = os.path.join(BASE_DIR, "data/features/all_factors.parquet")
    SCORES_PATH  = os.path.join(BASE_DIR, "data/models/composite_scores.parquet")
    OUTPUT_DIR   = os.path.join(BASE_DIR, "data/models")

    explanation_df = run(
        model_path       = MODEL_PATH,
        all_factors_path = FACTORS_PATH,
        scores_path      = SCORES_PATH,
        output_dir       = OUTPUT_DIR,
    )

    print("\n" + "─" * 70)
    print("TOP 10 LONG — SHAP EXPLANATIONS")
    print("─" * 70)
    long_df = explanation_df[explanation_df["recommendation"] == "LONG"].head(10)
    for _, row in long_df.iterrows():
        print(f"\n  {row['ticker']:<15} score={row.get('composite_alpha_score', 'N/A')}")
        print(f"  Driver 1: {row['top_factor_1']:<30} SHAP={row['shap_1']:+.5f}")
        print(f"  Driver 2: {row['top_factor_2']:<30} SHAP={row['shap_2']:+.5f}")
        print(f"  Driver 3: {row['top_factor_3']:<30} SHAP={row['shap_3']:+.5f}")
        print(f"  → {row['explanation']}")

    print("\n" + "─" * 70)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("─" * 70)
    global_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "shap_global_summary.parquet"))
    print(global_df.to_string(index=False))