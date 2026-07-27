"""Tests that every primitive has a consistent signature and is NaN-tolerant (P1-T2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphalineage.core.evaluate import evaluate, evaluate_python
from alphalineage.core.primitives import EPHEMERALS, OPERAND_FIELDS, OPERANDS, OPERATORS
from alphalineage.core.tree import InvalidTree, Node, validate
from alphalineage.core.types import DType


def _arg_node(dtype: DType) -> Node:
    # 'returns' carries a leading NaN row, so every operator is exercised on NaN input.
    if dtype in {DType.SERIES, DType.SIGNAL}:
        return Node("returns")
    if dtype is DType.SCALAR:
        return Node("const", value=2.0)
    if dtype is DType.WINDOW:
        return Node("window", value=3)
    if dtype is DType.BOOL:
        return Node("gt", (Node("close"), Node("open")))
    raise AssertionError(dtype)


def _operator_arg(primitive, index: int, dtype: DType) -> Node:
    inputs = (primitive.macro_policy or {}).get("inputs") or []
    item = inputs[index] if index < len(inputs) and isinstance(inputs[index], dict) else {}
    default = item.get("default")
    if default is not None and dtype is DType.WINDOW:
        return Node("window", value=int(default))
    if default is not None and dtype is DType.SCALAR:
        return Node("const", value=float(default))
    return _arg_node(dtype)


def test_operator_signatures_consistent():
    for prim in OPERATORS.values():
        assert prim.arity == len(prim.arg_types)
        # Built-ins execute a function; saved/catalog formulas execute their typed macro body.
        assert (prim.fn is not None) != (prim.macro_body is not None)
        assert prim.out_type in (DType.SERIES, DType.SIGNAL, DType.BOOL)


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
        node = Node(
            prim.name,
            tuple(_operator_arg(prim, index, dtype) for index, dtype in enumerate(prim.arg_types)),
        )
        validate(node)
        result = evaluate(node, synthetic_panel)
        assert isinstance(result, pd.DataFrame), prim.name
        assert result.shape == expected_shape, prim.name


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "5"])
def test_window_literals_must_be_positive_integers(value):
    with pytest.raises(InvalidTree, match="positive integer"):
        validate(Node("window", value=value))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1.0"])
def test_scalar_literals_must_be_finite_numbers(value):
    with pytest.raises(InvalidTree, match="finite number"):
        validate(Node("const", value=value))  # type: ignore[arg-type]


def test_literal_resource_bounds_are_normalized_validation_errors():
    with pytest.raises(InvalidTree, match="window must be at most"):
        validate(Node("window", value=100_001))
    with pytest.raises(InvalidTree, match="finite number"):
        validate(Node("const", value=10**10_000))


def test_invalid_window_is_rejected_even_without_tree_validation(synthetic_panel):
    tree = Node("delay", (Node("close"), Node("window", value=0)))
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_python(tree, synthetic_panel)


def test_unknown_child_raises_invalid_tree_not_key_error():
    with pytest.raises(InvalidTree, match="unknown primitive 'bogus'"):
        validate(Node("rank", (Node("bogus"),)))


def test_indicator_smoothing_primitives_have_explicit_nan_and_seed_semantics():
    index = pd.RangeIndex(9)
    rma_source = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0, 6.0, 7.0, 8.0]}, index=index)
    rma = OPERATORS["ts_rma"].fn
    assert rma is not None
    actual_rma = rma(rma_source, 3)["A"].to_numpy()
    expected_rma = np.array(
        [np.nan, np.nan, 2.0, 8.0 / 3.0, np.nan, np.nan, np.nan, 6.0, 20.0 / 3.0]
    )
    assert np.allclose(actual_rma, expected_rma, equal_nan=True)

    smooth_source = pd.DataFrame({"A": [np.nan, 20.0, 30.0, np.nan, 60.0]})
    smooth = OPERATORS["ts_recursive_smooth"].fn
    assert smooth is not None
    actual_smooth = smooth(smooth_source, 3, 50.0)["A"].to_numpy()
    expected_smooth = np.array([np.nan, 40.0, 110.0 / 3.0, np.nan, 400.0 / 9.0])
    assert np.allclose(actual_smooth, expected_smooth, equal_nan=True)

    population_std = OPERATORS["ts_std_pop"].fn
    assert population_std is not None
    actual_std = population_std(pd.DataFrame({"A": [1.0, 2.0, 3.0]}), 3).iloc[-1, 0]
    assert actual_std == pytest.approx(math.sqrt(2.0 / 3.0))


def test_cumulative_sum_restarts_after_non_finite_gaps():
    cumulative = OPERATORS["ts_cumsum"].fn
    assert cumulative is not None
    source = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, -1.0, np.inf, 3.0]})
    actual = cumulative(source)["A"].to_numpy()
    expected = np.array([1.0, 3.0, np.nan, 4.0, 3.0, np.nan, 3.0])
    assert np.allclose(actual, expected, equal_nan=True)
