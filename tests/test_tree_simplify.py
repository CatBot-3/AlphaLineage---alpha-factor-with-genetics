"""Tests for the algebraic simplifier (P1-T6): identities reduce and preserve semantics."""

from __future__ import annotations

import pandas as pd

from alphalineage.core.evaluate import evaluate
from alphalineage.core.simplify import simplify
from alphalineage.core.tree import Node


def test_scalar_identities():
    assert simplify(Node("add_scalar", (Node("close"), Node("const", value=0.0)))) == Node("close")
    assert simplify(Node("mul_scalar", (Node("close"), Node("const", value=1.0)))) == Node("close")
    assert simplify(Node("signed_power", (Node("close"), Node("const", value=1.0)))) == Node(
        "close"
    )


def test_window_identities():
    assert simplify(Node("delay", (Node("close"), Node("window", value=0)))) == Node("close")
    assert simplify(Node("ts_mean", (Node("close"), Node("window", value=1)))) == Node("close")
    assert simplify(Node("ts_sum", (Node("close"), Node("window", value=1)))) == Node("close")


def test_unary_idempotents_and_double_negation():
    assert simplify(Node("neg", (Node("neg", (Node("close"),)),))) == Node("close")
    assert simplify(Node("abs", (Node("abs", (Node("close"),)),))) == Node("abs", (Node("close"),))
    assert simplify(Node("abs", (Node("neg", (Node("close"),)),))) == Node("abs", (Node("close"),))
    assert simplify(Node("sign", (Node("sign", (Node("close"),)),))) == Node(
        "sign", (Node("close"),)
    )


def test_nested_simplification_to_fixpoint():
    # add_scalar(ts_mean(returns, 1), 0) -> returns
    tree = Node(
        "add_scalar",
        (Node("ts_mean", (Node("returns"), Node("window", value=1))), Node("const", value=0.0)),
    )
    assert simplify(tree) == Node("returns")


def test_simplify_preserves_semantics(synthetic_panel):
    tree = Node(
        "add_scalar",
        (Node("ts_mean", (Node("returns"), Node("window", value=1))), Node("const", value=0.0)),
    )
    pd.testing.assert_frame_equal(
        evaluate(tree, synthetic_panel), evaluate(simplify(tree), synthetic_panel)
    )
