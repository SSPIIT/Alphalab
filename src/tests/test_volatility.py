import pandas as pd
import numpy as np
import pytest

from src.factors.volatility import (
    compute_volatility,
    rank_volatility,
    composite_volatility_score
)


def create_ohlcv():
    n_days = 200
    dates = pd.date_range(end="2024-01-01", periods=n_days)

    df = pd.DataFrame({
        "ticker": ["A"] * n_days + ["B"] * n_days,
        "Date": list(dates) * 2,
        "Open": list(np.linspace(100, 120, n_days)) + list(np.linspace(200, 180, n_days)),
        "High": list(np.linspace(101, 121, n_days)) + list(np.linspace(201, 181, n_days)),
        "Low": list(np.linspace(99, 119, n_days)) + list(np.linspace(199, 179, n_days)),
        "Close": list(np.linspace(100, 120, n_days)) + list(np.linspace(200, 180, n_days)),
    })
    return df


def create_index():
    n_days = 200
    dates = pd.date_range(end="2024-01-01", periods=n_days)

    return pd.DataFrame({
        "ticker": ["INDEX"] * n_days,
        "Date": dates,
        "Close": np.linspace(1000, 1100, n_days)
    })


# ✅ Basic computation
def test_compute_volatility_basic():
    df = create_ohlcv()

    result = compute_volatility(df)

    assert "hist_vol_6m" in result.columns
    assert "atr_pct" in result.columns
    assert len(result) == 2


# ✅ With index (beta test)
def test_compute_volatility_with_index():
    df = create_ohlcv()
    index_df = create_index()

    result = compute_volatility(df, index_df)

    assert "beta" in result.columns


# ✅ Not enough history
def test_not_enough_history():
    df = create_ohlcv().iloc[:50]  # less than 126

    with pytest.raises(ValueError):
        compute_volatility(df)


# ✅ Downside volatility logic
def test_downside_volatility():
    df = create_ohlcv()

    # force negative returns
    df.loc[df.index[-50:], "Close"] = np.linspace(120, 80, 50)

    result = compute_volatility(df)

    assert "downside_vol_6m" in result.columns


# ✅ Ranking
def test_rank_volatility():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "hist_vol_1m": [0.3, 0.2, 0.1],
        "hist_vol_3m": [0.3, 0.2, 0.1],
        "hist_vol_6m": [0.3, 0.2, 0.1],
        "atr_pct": [0.03, 0.02, 0.01],
        "downside_vol_6m": [0.3, 0.2, 0.1],
        "beta": [1.5, 1.0, 0.5],
        "max_drawdown_6m": [-0.3, -0.2, -0.1],
    })

    ranked = rank_volatility(df)

    assert "hist_vol_6m_rank" in ranked.columns
    assert ranked["hist_vol_6m_rank"].max() == 1.0


# ✅ Composite score
def test_composite_volatility_score():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "hist_vol_1m_rank": [1.0, 0.0],
        "hist_vol_3m_rank": [1.0, 0.0],
    })

    result = composite_volatility_score(df)

    assert "volatility_composite" in result.columns
    assert result["volatility_composite"].max() == 1.0