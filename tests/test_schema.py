"""Tests for the canonical schema normalization."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphaforge.data.schema import PRICE_COLUMNS, normalize, validate


def test_normalize_tz_columns_and_defaults():
    # tz-aware, unsorted, missing corporate-action columns.
    idx = pd.to_datetime(["2020-01-02T00:00:00Z", "2020-01-01T00:00:00Z"])
    df = pd.DataFrame(
        {"open": [2, 1], "high": [2, 1], "low": [2, 1], "close": [2, 1], "volume": [10, 20]},
        index=idx,
    )

    out = normalize(df)

    assert list(out.columns) == PRICE_COLUMNS
    assert out.index.tz is None
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing  # sorted ascending
    assert out["div_cash"].tolist() == [0.0, 0.0]  # default filled
    assert out["split_factor"].tolist() == [1.0, 1.0]  # default filled
    assert str(out["volume"].dtype) == "float64"


def test_normalize_is_idempotent():
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {c: range(1, 6) for c in ["open", "high", "low", "close", "volume"]}, index=idx
    )
    once = normalize(df)
    pd.testing.assert_frame_equal(once, normalize(once))


def test_validate_rejects_bad_frame():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"close": [1.0]}))


@given(n=st.integers(min_value=1, max_value=50))
def test_normalize_dedups_and_sorts(n: int):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    # Duplicate the last row's date to exercise dedup (keep last).
    dup_idx = idx.append(pd.DatetimeIndex([idx[-1]]))
    values = list(range(1, n + 2))
    df = pd.DataFrame(
        {c: values for c in ["open", "high", "low", "close", "volume"]}, index=dup_idx
    )
    out = normalize(df)
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert len(out) == n
