"""Tests that every primitive has a consistent signature and is NaN-tolerant (P1-T2)."""

from __future__ import annotations

import pandas as pd

from alphaforge.core.evaluate import evaluate
from alphaforge.core.primitives import EPHEMERALS, OPERAND_FIELDS, OPERANDS, OPERATORS
from alphaforge.core.tree import Node, validate
from alphaforge.core.types import DType


def _arg_node(dtype: DType) -> Node:
    # 'returns' carries a leading NaN row, so every operator is exercised on NaN input.
    if dtype is DType.SERIES:
        return Node("returns")
    if dtype is DType.SCALAR:
        return Node("const", value=2.0)
    if dtype is DType.WINDOW:
        return Node("window", value=3)
    raise AssertionError(dtype)


def test_operator_signatures_consistent():
    for prim in OPERATORS.values():
        assert prim.arity == len(prim.arg_types)
        assert prim.fn is not None
        assert prim.out_type in (DType.SERIES, DType.SIGNAL)


def test_operands_and_ephemerals_consistent():
    assert set(OPERANDS) == set(OPERAND_FIELDS)
    for prim in OPERANDS.values():
        assert prim.out_type is DType.SERIES and prim.panel_field is not None
    assert EPHEMERALS["const"].out_type is DType.SCALAR
    assert EPHEMERALS["window"].out_type is DType.WINDOW
    for prim in EPHEMERALS.values():
        assert prim.sampler is not None


def test_every_operator_evaluates_without_error_on_nan_input(synthetic_panel):
    expected_shape = synthetic_panel["close"].shape
    for prim in OPERATORS.values():
        node = Node(prim.name, tuple(_arg_node(t) for t in prim.arg_types))
        validate(node)
        result = evaluate(node, synthetic_panel)
        assert isinstance(result, pd.DataFrame), prim.name
        assert result.shape == expected_shape, prim.name
