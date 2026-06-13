"""Tests for the optional C++ evaluator backend.

The flatten/IR and the Python-fallback dispatch are exercised always (no compiler needed). The
C++≡Python parity and fallback-on-unsupported tests run only when the extension is built.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from alphalineage.core import cpp
from alphalineage.core.evaluate import evaluate, evaluate_python
from alphalineage.core.generate import RandomTreeGenerator
from alphalineage.core.tree import Node

# Parity bar: bit-exact equality is impossible across C++/pandas float ordering, so the two
# backends are pinned to agree within a tight tolerance (~8 significant figures).
_RTOL, _ATOL = 1e-7, 1e-9


# --- always run (no compiler required) ------------------------------------------
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
    assert instrs[root][0] == cpp.CPP_OPCODES["rank"]  # root is the rank op
    assert cpp.OP_LOAD in opcodes  # operand loads
    assert {cpp.CPP_OPCODES["ts_mean"], cpp.CPP_OPCODES["sub"]} <= set(opcodes)

    # an unsupported op makes the whole tree non-flattenable (=> Python fallback)
    unsupported = Node("ts_corr", (Node("close"), Node("returns"), Node("window", value=5)))
    assert cpp.flatten(unsupported) is None


def test_dispatch_python_matches_baseline(synthetic_panel, monkeypatch):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    gen = RandomTreeGenerator(random.Random(3), max_depth=4, max_nodes=20)
    for tree in gen.ramped_half_and_half(200, min_depth=2, max_depth=4):
        pd.testing.assert_frame_equal(
            evaluate(tree, synthetic_panel), evaluate_python(tree, synthetic_panel)
        )


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    assert cpp.backend_enabled() is False
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    assert cpp.backend_enabled() == cpp.available()


# --- run only when the C++ extension is built -----------------------------------
@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
def test_cpp_python_parity(synthetic_panel):
    gen = RandomTreeGenerator(random.Random(0), max_depth=5, max_nodes=30)
    checked = 0
    for tree in gen.ramped_half_and_half(500, min_depth=2, max_depth=5):
        cpp_result = cpp.evaluate_cpp(tree, synthetic_panel)
        if cpp_result is None:  # unsupported op -> would fall back to Python
            continue
        checked += 1
        py_result = evaluate_python(tree, synthetic_panel)
        assert np.allclose(
            cpp_result.to_numpy(), py_result.to_numpy(), equal_nan=True, rtol=_RTOL, atol=_ATOL
        ), str(tree)
    assert checked > 50  # the backend actually handled a meaningful share of trees


@pytest.mark.skipif(not cpp.available(), reason="C++ evaluator extension not built")
def test_unsupported_op_falls_back(synthetic_panel, monkeypatch):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "cpp")
    tree = Node("ts_rank", (Node("close"), Node("window", value=5)))  # not C++-supported
    assert cpp.evaluate_cpp(tree, synthetic_panel) is None
    pd.testing.assert_frame_equal(
        evaluate(tree, synthetic_panel), evaluate_python(tree, synthetic_panel)
    )
