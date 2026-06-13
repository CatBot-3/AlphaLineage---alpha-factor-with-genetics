"""Tests for the DSL type lattice (P1-T1)."""

from __future__ import annotations

from alphalineage.core.types import DType, is_subtype


def test_subtype_is_reflexive():
    for t in DType:
        assert is_subtype(t, t)


def test_series_and_signal_are_mutually_substitutable():
    assert is_subtype(DType.SERIES, DType.SIGNAL)
    assert is_subtype(DType.SIGNAL, DType.SERIES)


def test_scalar_and_window_are_not_panels():
    assert not is_subtype(DType.SCALAR, DType.SERIES)
    assert not is_subtype(DType.WINDOW, DType.SIGNAL)
    assert not is_subtype(DType.SERIES, DType.SCALAR)
    assert not is_subtype(DType.SCALAR, DType.WINDOW)
