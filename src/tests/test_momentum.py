import pandas as pd
import numpy as np
import pytest

from src.factors.momentum import compute_momentum, rank_momentum


def create_price_df(ticker="A", n_days=300, start_price=100, end_price=200):
    dates = pd.date_range(end="2024-01-01", periods=n_days)
    prices = np.linspace(start_price, end_price, n_days)

    return pd.DataFrame({
        "ticker": [ticker] * n_days,
        "Date": dates,
        "Close": prices
    })


def test_compute_momentum_basic():
    df = create_price_df()

    result = compute_momentum(df)

    assert "mom_1m" in result.columns
    assert "mom_3m" in result.columns
    assert "mom_6m" in result.columns
    assert "mom_12m" in result.columns

    assert result["mom_12m"].iloc[0] > 0


def test_compute_momentum_exact_value():
    df = create_price_df()

    df.loc[df.index[-1], "Close"] = 110
    df.loc[df.index[-21], "Close"] = 100

    result = compute_momentum(df)

    expected = (110 - 100) / 100
    assert abs(result["mom_1m"].iloc[0] - expected) < 1e-6


def test_not_enough_data():
    df = create_price_df(n_days=100)

    with pytest.raises(ValueError):
        compute_momentum(df)


def test_zero_price_handling():
    df = create_price_df()

    df.loc[df.index[-21], "Close"] = 0

    result = compute_momentum(df)

    assert np.isnan(result["mom_1m"].iloc[0])


def test_multiple_tickers():
    df1 = create_price_df("A", start_price=100, end_price=200)
    df2 = create_price_df("B", start_price=200, end_price=100)

    df = pd.concat([df1, df2])

    result = compute_momentum(df)

    assert len(result) == 2


def test_rank_momentum():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "mom_1m": [0.1, 0.2, 0.3],
        "mom_3m": [0.1, 0.2, 0.3],
        "mom_6m": [0.1, 0.2, 0.3],
        "mom_12m": [0.1, 0.2, 0.3],
    })

    ranked = rank_momentum(df)

    assert "mom_1m_rank" in ranked.columns
    assert ranked["mom_1m_rank"].min() >= 0
    assert ranked["mom_1m_rank"].max() <= 1