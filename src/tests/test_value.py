import pandas as pd
import numpy as np
import pytest

from src.factors.value import compute_value, rank_value, composite_value_score


def create_ohlcv():
    dates = pd.date_range(end="2024-01-01", periods=10)

    return pd.DataFrame({
        "ticker": ["A"] * 10 + ["B"] * 10,
        "Date": list(dates) * 2,
        "Close": list(np.linspace(100, 110, 10)) + list(np.linspace(200, 180, 10))
    })


def create_fundamentals():
    return pd.DataFrame({
        "ticker": ["A", "B"],
        "eps": [10, 5],
        "book_value_per_share": [50, 40],
        "ebitda": [100, 80],
        "enterprise_value": [1000, 900],
        "dividend_per_share": [2, 1],
    })


# ✅ Basic computation test
def test_compute_value_basic():
    ohlcv = create_ohlcv()
    fundamentals = create_fundamentals()

    result = compute_value(ohlcv, fundamentals)

    assert "pe_ratio" in result.columns
    assert "earnings_yield" in result.columns
    assert len(result) == 2


# ✅ Exact value test (P/E + earnings yield)
def test_compute_value_exact():
    ohlcv = create_ohlcv()
    fundamentals = create_fundamentals()

    result = compute_value(ohlcv, fundamentals)

    # For ticker A
    latest_price = 110
    eps = 10

    expected_pe = latest_price / eps
    expected_ey = eps / latest_price

    row = result[result["ticker"] == "A"].iloc[0]

    assert abs(row["pe_ratio"] - expected_pe) < 1e-4
    assert abs(row["earnings_yield"] - expected_ey) < 1e-6


# ✅ Missing fundamentals → should not crash
def test_missing_fundamentals():
    ohlcv = create_ohlcv()

    result = compute_value(ohlcv, None)

    assert "pe_ratio" in result.columns
    assert result["pe_ratio"].isna().all()


# ✅ Negative / zero EPS → NaN
def test_negative_eps():
    ohlcv = create_ohlcv()

    fundamentals = pd.DataFrame({
        "ticker": ["A", "B"],
        "eps": [-10, 0],  # invalid
    })

    result = compute_value(ohlcv, fundamentals)

    assert result["pe_ratio"].isna().all()


# ✅ Ranking test
def test_rank_value():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "pe_ratio": [10, 20, 30],
        "earnings_yield": [0.1, 0.05, 0.03],
        "pb_ratio": [1, 2, 3],
        "book_to_market": [1, 0.5, 0.3],
        "ev_to_ebitda": [5, 10, 15],
        "dividend_yield": [0.02, 0.01, 0.005],
    })

    ranked = rank_value(df)

    assert "pe_ratio_rank" in ranked.columns
    assert ranked["pe_ratio_rank"].max() == 1.0


# ✅ Composite score test
def test_composite_value_score():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "pe_ratio_rank": [1.0, 0.0],
        "earnings_yield_rank": [1.0, 0.0],
    })

    result = composite_value_score(df)

    assert "value_composite" in result.columns
    assert result["value_composite"].max() == 1.0