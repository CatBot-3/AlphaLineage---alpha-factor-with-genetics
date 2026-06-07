"""Phase 7 acceptance + supporting tests for user extensibility (tests/test_extensibility.py)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.core import extensions
from alphaforge.core.evaluate import evaluate
from alphaforge.core.extensions import (
    ARG,
    InvalidOperator,
    operator_counts,
    register_operator,
)
from alphaforge.core.generate import RandomTreeGenerator
from alphaforge.core.gp import GP, GPConfig
from alphaforge.core.primitives import OPERATORS, REGISTRY, operators_for
from alphaforge.core.tree import Node, is_valid, validate
from alphaforge.core.types import DType
from alphaforge.validation.deflated_sharpe import deflated_sharpe_ratio
from alphaforge.validation.trials import effective_trials


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot/restore the global registries so registration never leaks between tests."""
    ops, reg, users = dict(OPERATORS), dict(REGISTRY), dict(extensions.USER_OPERATORS)
    yield
    for table, saved in ((OPERATORS, ops), (REGISTRY, reg), (extensions.USER_OPERATORS, users)):
        table.clear()
        table.update(saved)


def _momentum_body() -> Node:
    # sub(ts_mean($arg0, $arg1), $arg0)
    return Node(
        "sub",
        (Node("ts_mean", (Node(ARG, value=0), Node(ARG, value=1))), Node(ARG, value=0)),
    )


def test_user_operator_valid_immediately(synthetic_panel):
    register_operator("momentum", [DType.SERIES, DType.WINDOW], DType.SERIES, _momentum_body())
    assert any(p.name == "momentum" for p in operators_for(DType.SERIES))

    # a hand-built tree using it validates and evaluates to the manual expansion
    tree = Node("momentum", (Node("close"), Node("window", value=5)))
    validate(tree)
    manual = Node("sub", (Node("ts_mean", (Node("close"), Node("window", value=5))), Node("close")))
    pd.testing.assert_frame_equal(
        evaluate(tree, synthetic_panel), evaluate(manual, synthetic_panel)
    )

    # 1k random trees are all type-valid, evaluate without error, and actually use the operator
    gen = RandomTreeGenerator(random.Random(0), max_depth=4, max_nodes=25)
    used = False
    for candidate in gen.ramped_half_and_half(1000, min_depth=2, max_depth=4):
        assert is_valid(candidate)
        assert isinstance(evaluate(candidate, synthetic_panel), pd.DataFrame)
        used = used or any(n.name == "momentum" for n in candidate.iter_nodes())
    assert used

    # crossover + mutation stay valid with the user operator in the set
    gp = GP(GPConfig(max_depth=5, max_nodes=30, seed=1), synthetic_panel)
    for _ in range(1000):
        a = gp.generator.generate(grow=True)
        b = gp.generator.generate(grow=False)
        assert is_valid(gp._crossover(a, b))
        assert is_valid(gp._subtree_mutation(a))
        assert is_valid(gp._point_mutation(a))


def test_no_arbitrary_code_path():
    # a body naming anything that is not an existing primitive is rejected — never executed
    with pytest.raises(InvalidOperator):
        register_operator("evil", [DType.SERIES], DType.SERIES, {"name": "__import__"})
    with pytest.raises(InvalidOperator):
        register_operator(
            "evil2",
            [DType.SERIES],
            DType.SERIES,
            {"name": "os.system", "children": [{"name": ARG, "value": 0}]},
        )

    # the registration + evaluation path contains no code-execution builtins
    core_dir = Path(extensions.__file__).parent
    for filename in ("extensions.py", "evaluate.py"):
        source = (core_dir / filename).read_text(encoding="utf-8")
        for forbidden in ("eval(", "exec(", "compile(", "__import__("):
            assert forbidden not in source, f"{forbidden} found in {filename}"


def test_trial_count_updates():
    builtin, user = operator_counts()
    assert user == 0
    base = effective_trials(1000, n_operators=builtin, baseline=builtin)

    register_operator(
        "m1",
        [DType.SERIES, DType.WINDOW],
        DType.SERIES,
        Node("ts_mean", (Node(ARG, value=0), Node(ARG, value=1))),
    )
    builtin2, user2 = operator_counts()
    assert user2 == 1 and builtin2 == builtin
    more = effective_trials(1000, n_operators=builtin2 + user2, baseline=builtin)

    assert more > base  # adding a primitive grows the effective trial count
    returns = pd.Series(np.random.default_rng(0).normal(0.05, 1.0, 250))
    # ... and that shows up as a (weakly) lower deflated Sharpe
    assert deflated_sharpe_ratio(returns, more, 0.25) <= deflated_sharpe_ratio(returns, base, 0.25)


def test_expand_and_registration_guards():
    register_operator(
        "spread",
        [DType.SERIES, DType.SERIES],
        DType.SERIES,
        Node("sub", (Node(ARG, value=0), Node(ARG, value=1))),
    )
    node = Node("spread", (Node("high"), Node("low")))
    expanded = extensions.expand(node, REGISTRY["spread"].macro_body)
    assert expanded == Node("sub", (Node("high"), Node("low")))

    # cannot clobber a built-in, and a wrong-output-type body is rejected
    with pytest.raises(InvalidOperator):
        register_operator("rank", [DType.SERIES], DType.SERIES, Node(ARG, value=0))
    with pytest.raises(InvalidOperator):
        register_operator("bad", [DType.SERIES], DType.WINDOW, Node(ARG, value=0))

    extensions.unregister_operator("spread")
    assert "spread" not in REGISTRY and "spread" not in OPERATORS
