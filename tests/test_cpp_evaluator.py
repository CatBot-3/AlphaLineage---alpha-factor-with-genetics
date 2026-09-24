"""Tests for the optional C++ evaluator backend and ordered batch dispatch."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from alphalineage.core import cpp
from alphalineage.core.evaluate import evaluate, evaluate_python
from alphalineage.core.fitness import forward_returns, score_tree
from alphalineage.core.generate import RandomTreeGenerator
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERATORS
from alphalineage.core.tree import Node

# Bit equality is not portable across pandas/compiler floating-point ordering. The native and
# Python implementations must nevertheless agree to roughly eight significant figures.
_RTOL, _ATOL = 1e-7, 1e-9


def _builtin_operator_names() -> set[str]:
    """Keep primitive-kernel tests independent of process-global saved/catalog formulas."""
    return {name for name, primitive in OPERATORS.items() if primitive.macro_body is None}


def test_flatten_ir():
    tree = Node(
        "rank",
        (
            Node(
                "sub", (Node("ts_mean", (Node("returns"), Node("window", value=5))), Node("close"))
            ),
        ),
    )
    plan = cpp.flatten(tree)
    assert plan is not None
    instrs, root = plan
    opcodes = [ins[0] for ins in instrs]
    assert instrs[root][0] == cpp.CPP_OPCODES["rank"]
    assert cpp.OP_LOAD in opcodes
    assert {cpp.CPP_OPCODES["ts_mean"], cpp.CPP_OPCODES["sub"]} <= set(opcodes)

    corr = Node("ts_corr", (Node("close"), Node("returns"), Node("window", value=5)))
    corr_plan = cpp.flatten(corr)
    assert corr_plan is not None
    assert corr_plan[0][corr_plan[1]][0] == cpp.CPP_OPCODES["ts_corr"]

    where = Node(
        "where",
        (Node("gt", (Node("close"), Node("open"))), Node("high"), Node("low")),
    )
    where_plan = cpp.flatten(where)
    assert where_plan is not None
    where_instruction = where_plan[0][where_plan[1]]
    assert where_instruction[0] == cpp.CPP_OPCODES["where"]
    assert all(dependency >= 0 for dependency in where_instruction[1:4])


def test_dispatch_python_matches_baseline(synthetic_panel, monkeypatch):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    gen = RandomTreeGenerator(
        random.Random(3),
        max_depth=4,
        max_nodes=20,
        allowed_operators=_builtin_operator_names(),
    )
    for tree in gen.ramped_half_and_half(200, min_depth=2, max_depth=4):
        pd.testing.assert_frame_equal(
            evaluate(tree, synthetic_panel), evaluate_python(tree, synthetic_panel)
        )


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    assert cpp.backend_enabled() is False
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    assert cpp.backend_enabled() == cpp.available()


def test_older_native_ema_uses_python_for_pandas_two(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cpp, "_EXT", SimpleNamespace(ABI_VERSION=9))
    monkeypatch.setattr(cpp, "_EMA_DECAYED_NEW_WEIGHT", False)
    tree = Node("ts_ema", (Node("close"), Node("window", value=3)))
    assert not cpp._plan_supported_by_loaded_abi(cpp._compile(tree))
    assert cpp._plan_supported_by_loaded_abi(cpp._compile(Node("close")))


def _expanded_operator_trees() -> list[Node]:
    window = Node("window", value=5)
    close, open_, high, low = (Node(name) for name in ("close", "open", "high", "low"))
    greater = Node("gt", (close, open_))
    less = Node("lt", (low, high))
    return [
        Node("ts_ema", (close, window)),
        Node("ts_rma", (close, window)),
        Node("ts_std_pop", (close, window)),
        Node("ts_recursive_smooth", (close, window, Node("const", value=50.0))),
        Node("ts_rank", (close, window)),
        Node("decay_linear", (close, window)),
        Node("ts_corr", (close, open_, window)),
        Node("ts_cov", (close, open_, window)),
        Node("scale", (close,)),
        greater,
        Node("ge", (close, open_)),
        less,
        Node("le", (low, high)),
        Node("and_", (greater, less)),
        Node("or_", (greater, less)),
        Node("not_", (greater,)),
        Node("where", (greater, high, low)),
    ]


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
def test_cpp_python_parity(synthetic_panel):
    gen = RandomTreeGenerator(
        random.Random(0),
        max_depth=5,
        max_nodes=30,
        allowed_operators=_builtin_operator_names(),
    )
    checked = 0
    for tree in gen.ramped_half_and_half(500, min_depth=2, max_depth=5):
        cpp_result = cpp.evaluate_cpp(tree, synthetic_panel)
        if cpp_result is None:
            continue
        checked += 1
        py_result = evaluate_python(tree, synthetic_panel)
        assert np.allclose(
            cpp_result.to_numpy(), py_result.to_numpy(), equal_nan=True, rtol=_RTOL, atol=_ATOL
        ), str(tree)
    assert checked == 500


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
@pytest.mark.parametrize("tree", _expanded_operator_trees(), ids=str)
def test_new_native_operator_parity(synthetic_panel, tree):
    actual = cpp.evaluate_cpp(tree, synthetic_panel)
    assert actual is not None
    expected = evaluate_python(tree, synthetic_panel)
    if tree.out_type.value == "bool":
        pd.testing.assert_frame_equal(actual, expected)
    else:
        assert np.allclose(
            actual.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=_RTOL, atol=_ATOL
        )


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
def test_native_parity_with_nan_infinity_and_ties(synthetic_panel):
    fields = {name: frame.copy() for name, frame in synthetic_panel.fields.items()}
    close = fields["close"]
    close.iloc[5, :] = 100.0  # a complete cross-sectional tie
    close.iloc[8, 1] = np.nan
    close.iloc[12, 2] = np.inf
    close.iloc[16, 3] = -np.inf
    panel = Panel(fields)
    window = Node("window", value=3)
    trees = [
        Node("rank", (Node("close"),)),
        Node("zscore", (Node("close"),)),
        Node("scale", (Node("close"),)),
        Node("ts_mean", (Node("close"), window)),
        Node("ts_std", (Node("close"), window)),
        Node("ts_std_pop", (Node("close"), window)),
        Node("ts_rma", (Node("close"), window)),
        Node(
            "ts_recursive_smooth",
            (Node("close"), window, Node("const", value=50.0)),
        ),
        Node("ts_min", (Node("close"), window)),
        Node("ts_max", (Node("close"), window)),
        Node("ts_ema", (Node("close"), window)),
        Node("ts_rank", (Node("close"), window)),
        Node("decay_linear", (Node("close"), window)),
        Node("ts_cov", (Node("close"), Node("open"), window)),
        Node("ts_corr", (Node("close"), Node("open"), window)),
    ]
    for tree in trees:
        actual = cpp.evaluate_cpp(tree, panel)
        assert actual is not None
        expected = evaluate_python(tree, panel)
        assert np.allclose(
            actual.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=_RTOL, atol=_ATOL
        ), str(tree)


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
def test_evaluate_many_is_ordered_worker_invariant_and_memory_bounded(
    synthetic_panel, monkeypatch
):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    trees = _expanded_operator_trees() + [Node("rank", (Node("returns"),))]
    serial = cpp.evaluate_many(trees, synthetic_panel, workers=1)
    parallel = cpp.evaluate_many(trees, synthetic_panel, workers=4)
    constrained = cpp.evaluate_many(trees, synthetic_panel, workers=4, memory_budget_bytes=1)
    assert all(frame is not None for frame in serial)
    for expected, actual, limited in zip(serial, parallel, constrained, strict=True):
        assert expected is not None and actual is not None and limited is not None
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(limited, expected)


def test_evaluate_many_returns_fallback_markers_when_native_disabled(synthetic_panel, monkeypatch):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    trees = [Node("close"), Node("rank", (Node("close"),))]
    assert cpp.evaluate_many(trees, synthetic_panel, workers=4) == [None, None]


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
@pytest.mark.parametrize("absolute", [True, False])
def test_native_score_many_parity_and_worker_invariance(signal_panel, monkeypatch, absolute):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    panel, _ = signal_panel
    target = forward_returns(panel)
    generator = RandomTreeGenerator(
        random.Random(17),
        max_depth=5,
        max_nodes=25,
        allowed_operators=_builtin_operator_names(),
    )
    trees = generator.ramped_half_and_half(60, min_depth=2, max_depth=5)
    expected = [
        score_tree(
            tree,
            panel,
            target,
            method="spearman",
            absolute=absolute,
            parsimony=0.001,
            min_names=5,
        )
        for tree in trees
    ]
    serial = cpp.score_many(
        trees,
        panel,
        target,
        absolute=absolute,
        parsimony=0.001,
        workers=1,
    )
    parallel = cpp.score_many(
        trees,
        panel,
        target,
        absolute=absolute,
        parsimony=0.001,
        workers=4,
    )
    constrained = cpp.score_many(
        trees,
        panel,
        target,
        absolute=absolute,
        parsimony=0.001,
        workers=4,
        memory_budget_bytes=1,
    )
    assert serial == parallel == constrained
    for actual, reference in zip(serial, expected, strict=True):
        assert actual is not None
        assert np.isclose(actual[0], reference[0], rtol=1e-12, atol=1e-14)
        assert np.isclose(actual[1]["ic"], reference[1]["ic"], rtol=1e-12, atol=1e-14)
        assert np.isclose(actual[1]["ic_ir"], reference[1]["ic_ir"], rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
def test_native_macro_parsimony_uses_expanded_node_count(signal_panel, monkeypatch):
    from alphalineage.core.extensions import ARG, register_operator, unregister_operator
    from alphalineage.core.types import DType

    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    panel, _ = signal_panel
    target = forward_returns(panel)
    register_operator(
        "expanded_penalty_native_test",
        [DType.SERIES],
        DType.SERIES,
        Node(
            "add",
            (
                Node(ARG, value=0),
                Node("mul_scalar", (Node(ARG, value=0), Node("const", value=0.0))),
            ),
        ),
    )
    try:
        tree = Node("expanded_penalty_native_test", (Node("volume"),))
        expected = score_tree(tree, panel, target, parsimony=0.1)
        actual = cpp.score_many([tree], panel, target, parsimony=0.1, workers=2)[0]
        assert actual is not None
        assert np.isclose(actual[0], expected[0], rtol=1e-12, atol=1e-14)
        assert np.isclose(actual[1]["ic"], expected[1]["ic"], rtol=1e-12, atol=1e-14)
        assert np.isclose(
            actual[1]["ic_ir"], expected[1]["ic_ir"], rtol=1e-12, atol=1e-14
        )
    finally:
        unregister_operator("expanded_penalty_native_test")
        cpp.clear_plan_cache()


def test_native_scoring_capability_and_pearson_fallback(signal_panel, monkeypatch):
    panel, _ = signal_panel
    target = forward_returns(panel)
    tree = Node("rank", (Node("returns"),))
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    assert cpp.supports_native_scoring("spearman") == cpp.available()
    assert cpp.supports_native_scoring("pearson") is False
    assert cpp.score_many([tree], panel, target, method="pearson", workers=4) == [None]


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
def test_native_score_many_rank_mask_infinity_and_degenerate_parity(
    synthetic_panel, monkeypatch
):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    fields = {name: frame.iloc[:12].copy() for name, frame in synthetic_panel.fields.items()}
    patterns = np.array(
        [
            [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # average ties
            [1.0, np.nan, 2.0, 2.0, 4.0, 5.0],  # exactly five paired names
            [-np.inf, 0.0, 1.0, np.inf, 2.0, 3.0],  # infinities are ranked observations
            [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],  # below min_names=5
        ]
        * 3,
        dtype=np.float64,
    )
    fields["close"].iloc[:, :] = patterns
    fields["open"].iloc[:, :] = 7.0  # constant factor => zero denominator on every date
    panel = Panel(fields)
    target_patterns = np.array(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [np.inf, 3.0, 2.0, -np.inf, 1.0, 0.0],
            [4.0, 3.0, 2.0, 1.0, 6.0, 5.0],
        ]
        * 3,
        dtype=np.float64,
    )
    target = pd.DataFrame(target_patterns, index=panel.dates, columns=panel.symbols)
    trees = [
        Node("close"),
        Node("rank", (Node("close"),)),
        Node("ts_mean", (Node("close"), Node("window", value=2))),
        Node("open"),
    ]
    expected = [
        score_tree(tree, panel, target, parsimony=0.01, min_names=5) for tree in trees
    ]
    native = cpp.score_many(
        trees,
        panel,
        target,
        parsimony=0.01,
        min_names=5,
        min_valid_dates=5,
        workers=4,
    )
    for actual, reference in zip(native, expected, strict=True):
        assert actual is not None
        assert np.isclose(actual[0], reference[0], rtol=1e-12, atol=1e-14)
        assert np.isclose(actual[1]["ic"], reference[1]["ic"], rtol=1e-12, atol=1e-14)
        assert np.isclose(actual[1]["ic_ir"], reference[1]["ic_ir"], rtol=1e-12, atol=1e-14)

    # No row can pass a seven-name breadth floor, and only nine dates pass min_names=5.
    too_wide = cpp.score_many(
        [Node("close")], panel, target, parsimony=0.01, min_names=7, workers=2
    )
    too_short = cpp.score_many(
        [Node("close")],
        panel,
        target,
        parsimony=0.01,
        min_names=5,
        min_valid_dates=10,
        workers=2,
    )
    expected_degenerate = (-0.01, {"ic": 0.0, "ic_ir": 0.0})
    assert too_wide == [expected_degenerate]
    assert too_short == [expected_degenerate]


@pytest.mark.parametrize("workers", [0, -1, True])
def test_evaluate_many_rejects_invalid_worker_counts(synthetic_panel, workers):
    with pytest.raises(ValueError, match="workers"):
        cpp.evaluate_many([Node("close")], synthetic_panel, workers=workers)
