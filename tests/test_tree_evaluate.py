"""P1 acceptance test ``test_evaluator_matches_reference``: evaluator vs hand-computed factors."""

from __future__ import annotations

import pandas as pd

from alphaforge.core.evaluate import evaluate
from alphaforge.core.tree import Node


def test_evaluator_matches_reference(synthetic_panel):
    p = synthetic_panel

    # f1: cross-sectional percentile rank of close
    f1 = Node("rank", (Node("close"),))
    pd.testing.assert_frame_equal(evaluate(f1, p), p["close"].rank(axis=1, pct=True))

    # f2: 5-day moving average of returns
    f2 = Node("ts_mean", (Node("returns"), Node("window", value=5)))
    pd.testing.assert_frame_equal(evaluate(f2, p), p["returns"].rolling(5, min_periods=5).mean())

    # f3: one-day change in close
    f3 = Node("sub", (Node("close"), Node("delay", (Node("close"), Node("window", value=1)))))
    pd.testing.assert_frame_equal(evaluate(f3, p), p["close"] - p["close"].shift(1))
