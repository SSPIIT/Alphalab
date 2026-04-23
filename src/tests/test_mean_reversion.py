import pandas as pd
import numpy as np
import pytest

from src.factors.mean_reversion import (
    compute_mean_reversion,
    rank_mean_reversion,
    composite_mean_reversion_score
)


def create_ohlcv():
    n_days = 200
    dates = pd.date_range(end="2024-01-01", periods=n_days)

    df = pd.DataFrame({
        "ticker": ["A"] * n_days + ["B"] * n_days,
        "Date": list(dates) * 2,
        "Close": list(np.linspace(100, 120, n_days)) + list(np.linspace(200, 180, n_days)),
    })
    return df


# ✅ Basic computation
def test_compute_mean_reversion_basic():
    df = create_ohlcv()

    result = compute_mean_reversion(df)

    assert "rsi_14" in result.columns
    assert "zscore_20d" in result.columns
    assert "bb_pct_b" in result.columns
    assert len(result) == 2


# ✅ Exact logic (short-term reversal)
def test_short_term_reversal():
    df = create_ohlcv()

    # get indices for ticker A
    idx = df[df["ticker"] == "A"].index

    # FORCE a strong drop in last 21 days
    df.loc[idx[-21:], "Close"] = np.linspace(200, 50, 21)

    result = compute_mean_reversion(df)

    val = result[result["ticker"] == "A"]["short_term_reversal"].iloc[0]

    assert val > 0


# ✅ Not enough history
def test_not_enough_history():
    df = create_ohlcv().iloc[:30]

    with pytest.raises(ValueError):
        compute_mean_reversion(df)


# ✅ Z-score behavior
def test_zscore_behavior():
    df = create_ohlcv()

    # make price spike → high zscore
    df.loc[df.index[-1], "Close"] = 500

    result = compute_mean_reversion(df)

    assert result["zscore_20d"].iloc[0] > 0


# ✅ RSI bounds
def test_rsi_bounds():
    df = create_ohlcv()

    result = compute_mean_reversion(df)

    rsi = result["rsi_14"].iloc[0]
    assert 0 <= rsi <= 100


# ✅ Ranking
def test_rank_mean_reversion():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "rsi_14": [20, 50, 80],
        "zscore_20d": [-2, 0, 2],
        "zscore_60d": [-2, 0, 2],
        "bb_pct_b": [0.1, 0.5, 0.9],
        "dist_from_52w_high": [-0.5, -0.2, -0.1],
        "dist_from_52w_low": [0.1, 0.3, 0.6],
        "short_term_reversal": [0.2, 0.0, -0.2],
    })

    ranked = rank_mean_reversion(df)

    assert "rsi_14_rank" in ranked.columns
    assert ranked["rsi_14_rank"].max() == 1.0


# ✅ Composite score
def test_composite_mean_reversion_score():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "rsi_14_rank": [1.0, 0.0],
        "zscore_20d_rank": [1.0, 0.0],
    })

    result = composite_mean_reversion_score(df)

    assert "mean_reversion_composite" in result.columns
    assert result["mean_reversion_composite"].max() == 1.0