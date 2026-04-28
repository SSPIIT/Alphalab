import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# VALUE COMPUTATION
# ─────────────────────────────────────────────
def compute_value(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes value factor scores from pre-fetched fundamentals.

    INPUT:
        fundamentals_df with columns:
        [ticker, pe_ratio, pb_ratio, ps_ratio, ...]

    OUTPUT:
        value factor dataframe with ranks
    """

    df = fundamentals_df.copy()

    # ── Handle missing columns safely ─────────────────────────
    required_cols = ["pe_ratio", "pb_ratio", "ps_ratio"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # ── Winsorization (remove extreme outliers) ───────────────
    for col in required_cols:
        if df[col].notna().sum() > 0:
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = df[col].clip(lower=lower, upper=upper)

    # ── Invert ratios (lower = better → higher score) ─────────
    eps = 1e-6
    df["value_pe"] = 1 / (df["pe_ratio"] + eps)
    df["value_pb"] = 1 / (df["pb_ratio"] + eps)
    df["value_ps"] = 1 / (df["ps_ratio"] + eps)

    # ── Composite score ──────────────────────────────────────
    value_cols = ["value_pe", "value_pb", "value_ps"]
    df["value_composite"] = df[value_cols].mean(axis=1, skipna=True)

    # ── Ranking (percentile 0–1) ─────────────────────────────
    for col in value_cols + ["value_composite"]:
        df[f"{col}_rank"] = df[col].rank(pct=True, na_option="keep")

    logger.info(f"Value factors computed for {len(df)} stocks")
    logger.info(
        f"value_composite stats: mean={df['value_composite'].mean():.4f}, "
        f"std={df['value_composite'].std():.4f}"
    )

    # ── Final columns ────────────────────────────────────────
    output_cols = [
        "ticker",
        "pe_ratio", "pb_ratio", "ps_ratio",
        "value_pe", "value_pb", "value_ps", "value_composite",
        "value_pe_rank", "value_pb_rank", "value_ps_rank", "value_composite_rank"
    ]

    return df[output_cols]


# ─────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────
def run(fundamentals_path: str, output_path: str) -> pd.DataFrame:
    """
    Loads fundamentals parquet → computes value → saves output
    """

    logger.info(f"Loading fundamentals from {fundamentals_path}")

    if not os.path.exists(fundamentals_path):
        raise FileNotFoundError(f"File not found: {fundamentals_path}")

    fundamentals = pd.read_parquet(fundamentals_path)

    if fundamentals.empty:
        raise ValueError("Fundamentals data is empty")

    value_df = compute_value(fundamentals)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    value_df.to_parquet(output_path, index=False)

    logger.info(f"Value factors saved to {output_path}")

    return value_df


# ─────────────────────────────────────────────
# LOCAL TESTING
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    features_dir = "data/features"

    # find latest fundamentals file
    files = sorted([f for f in os.listdir(features_dir) if "fundamentals" in f])
    if not files:
        raise ValueError("No fundamentals files found")

    latest = os.path.join(features_dir, files[-1])

    result = run(
        fundamentals_path=latest,
        output_path="data/features/value.parquet"
    )

    print("\n── Top 10 Value Stocks (Cheapest) ──")
    print(
        result.sort_values("value_composite_rank", ascending=False)
        .head(10)[["ticker", "pe_ratio", "pb_ratio", "value_composite_rank"]]
        .to_string(index=False)
    )

    print("\n── Bottom 10 (Most Expensive) ──")
    print(
        result.sort_values("value_composite_rank", ascending=True)
        .head(10)[["ticker", "pe_ratio", "pb_ratio", "value_composite_rank"]]
        .to_string(index=False)
    )