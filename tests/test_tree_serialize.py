"""P1 acceptance: lossless JSON serialization round-trip (test_serialization_roundtrip)."""

from __future__ import annotations

import random

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from alphaforge.core import tree as T
from alphaforge.core.evaluate import evaluate
from alphaforge.core.generate import RandomTreeGenerator
from alphaforge.core.tree import Node


def test_roundtrip_handbuilt():
    tree = Node(
        "sub",
        (Node("close"), Node("delay", (Node("close"), Node("window", value=1)))),
    )
    assert T.from_json(T.to_json(tree)) == tree


def test_roundtrip_preserves_ephemeral_value_types():
    tree = Node("mul_scalar", (Node("close"), Node("const", value=2.0)))
    restored = T.from_json(T.to_json(tree))
    assert restored == tree
    assert isinstance(restored.children[1].value, float)


@given(seed=st.integers(min_value=0, max_value=10_000))
def test_roundtrip_generated_structural(seed: int):
    gen = RandomTreeGenerator(random.Random(seed), max_depth=5, max_nodes=30)
    tree = gen.generate(grow=(seed % 2 == 0))
    assert T.from_json(T.to_json(tree)) == tree


def test_roundtrip_evaluates_identically(synthetic_panel):
    gen = RandomTreeGenerator(random.Random(7), max_depth=5, max_nodes=30)
    for _ in range(50):
        tree = gen.generate(grow=True)
        restored = T.from_json(T.to_json(tree))
        pd.testing.assert_frame_equal(
            evaluate(tree, synthetic_panel), evaluate(restored, synthetic_panel)
        )
