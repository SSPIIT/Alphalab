import pandas as pd
import numpy as np
import pytest

from src.factors.combine_factors import (
    merge_factors,
    compute_overall_composite
)


# ✅ Create dummy factor data
def create_factor_dfs():
    momentum = pd.DataFrame({
        "ticker": ["A", "B"],
        "mom_6m_rank": [1.0, 0.0]
    })

    value = pd.DataFrame({
        "ticker": ["A", "C"],
        "value_composite": [0.8, 0.2]
    })

    volatility = pd.DataFrame({
        "ticker": ["A", "B"],
        "volatility_composite": [0.9, 0.1]
    })

    mean_reversion = pd.DataFrame({
        "ticker": ["A", "B"],
        "mean_reversion_composite": [0.7, 0.3]
    })

    return {
        "momentum": momentum,
        "value": value,
        "volatility": volatility,
        "mean_reversion": mean_reversion
    }


# ✅ Test merge (outer join behavior)
def test_merge_factors():
    factor_dfs = create_factor_dfs()

    merged = merge_factors(factor_dfs)

    # Should contain all tickers: A, B, C
    assert set(merged["ticker"]) == {"A", "B", "C"}

    # Check columns exist
    assert "mom_6m_rank" in merged.columns
    assert "value_composite" in merged.columns


# ✅ Test overall composite calculation
def test_compute_overall_composite():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "mom_6m_rank": [1.0, 0.0],
        "value_composite": [0.8, 0.2],
        "volatility_composite": [0.9, 0.1],
        "mean_reversion_composite": [0.7, 0.3],
    })

    result = compute_overall_composite(df)

    assert "overall_composite" in result.columns

    # A should rank higher than B
    assert result.loc[result["ticker"] == "A", "overall_composite"].iloc[0] > \
           result.loc[result["ticker"] == "B", "overall_composite"].iloc[0]


# ✅ Missing composite columns
def test_missing_composites():
    df = pd.DataFrame({
        "ticker": ["A", "B"]
    })

    result = compute_overall_composite(df)

    assert "overall_composite" in result.columns
    assert result["overall_composite"].isna().all()


# ✅ Partial composites (robustness)
def test_partial_composites():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "mom_6m_rank": [1.0, 0.0],  # only one factor available
    })

    result = compute_overall_composite(df)

    assert "overall_composite" in result.columns
    assert not result["overall_composite"].isna().all()


# ✅ Empty input
def test_empty_merge():
    with pytest.raises(ValueError):
        merge_factors({})