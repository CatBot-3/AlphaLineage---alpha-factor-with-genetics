"""P1 acceptance: random-tree validity and depth/size constraints (P1-T4)."""

from __future__ import annotations

import random

import pandas as pd

from alphaforge.core.evaluate import evaluate
from alphaforge.core.generate import RandomTreeGenerator
from alphaforge.core.panel import Panel
from alphaforge.core.tree import is_valid


def test_random_trees_all_valid(synthetic_panel):
    # Tiny panel keeps 10k evaluations fast; correctness of values is covered elsewhere.
    small = Panel({k: v.iloc[:20, :5] for k, v in synthetic_panel.fields.items()})
    gen = RandomTreeGenerator(random.Random(0), max_depth=4, max_nodes=25)

    trees = gen.ramped_half_and_half(10_000, min_depth=2, max_depth=4)
    assert len(trees) == 10_000
    for tree in trees:
        assert is_valid(tree)
        result = evaluate(tree, small)
        assert isinstance(result, pd.DataFrame)


def test_depth_size_constraints():
    gen = RandomTreeGenerator(random.Random(1), max_depth=5, max_nodes=40)
    for tree in gen.ramped_half_and_half(3000, min_depth=2, max_depth=5):
        assert tree.depth() <= 5
        assert tree.size() <= 40

    # Degenerate budget: a single-depth tree is exactly one terminal node.
    leaf = RandomTreeGenerator(random.Random(2), max_depth=1, max_nodes=40).generate()
    assert leaf.depth() == 1
    assert leaf.size() == 1
