import pandas as pd
import numpy as np
import pytest

from src.factors.value import compute_value


def create_fundamentals():
    return pd.DataFrame({
        "ticker": ["A", "B"],
        "pe_ratio": [10, 20],
        "pb_ratio": [1.5, 2.0],
        "ps_ratio": [2.0, 3.0],
    })


# ✅ Basic computation
def test_compute_value_basic():
    fundamentals = create_fundamentals()

    result = compute_value(fundamentals)

    assert "value_pe" in result.columns
    assert "value_pb" in result.columns
    assert "value_ps" in result.columns
    assert "value_composite" in result.columns

    assert len(result) == 2


# ✅ Exact value test (inverse logic)
def test_compute_value_exact():
    fundamentals = pd.DataFrame({
        "ticker": ["A"],
        "pe_ratio": [10],
        "pb_ratio": [2],
        "ps_ratio": [5],
    })

    result = compute_value(fundamentals)

    row = result.iloc[0]

    assert abs(row["value_pe"] - (1 / 10)) < 1e-6
    assert abs(row["value_pb"] - (1 / 2)) < 1e-6
    assert abs(row["value_ps"] - (1 / 5)) < 1e-6


# ✅ Missing columns → should fail
def test_missing_columns():
    fundamentals = pd.DataFrame({
        "ticker": ["A"],
        "pe_ratio": [10],
    })

    with pytest.raises(ValueError):
        compute_value(fundamentals)


# ✅ NaN handling
def test_nan_handling():
    fundamentals = pd.DataFrame({
        "ticker": ["A", "B"],
        "pe_ratio": [10, np.nan],
        "pb_ratio": [2, 3],
        "ps_ratio": [5, 6],
    })

    result = compute_value(fundamentals)

    assert not result.empty


# ✅ Ranking correctness
def test_ranking_range():
    fundamentals = create_fundamentals()

    result = compute_value(fundamentals)

    assert result["value_composite_rank"].min() >= 0
    assert result["value_composite_rank"].max() <= 1


# ✅ Composite score existence
def test_composite_exists():
    fundamentals = create_fundamentals()

    result = compute_value(fundamentals)

    assert "value_composite" in result.columns