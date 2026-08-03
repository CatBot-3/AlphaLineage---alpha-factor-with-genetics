"""P10-T2 - inner purged-split scoring.

The property under test is not "does it produce a number" but "does it produce a number without
reading anything it must not read". Every test that touches data asserts the split guard too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalineage.agent.context import AgentContext, truncate_panel
from alphalineage.agent.evaluate import (
    MIN_INNER_DATES,
    correlation_with_target,
    inner_split,
    score_candidate,
)
from alphalineage.agent.guards import GuardViolation, SplitGuard
from alphalineage.api.sessions import Boundaries
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node

BOUNDARIES = Boundaries(
    train_end="2016-01-01",
    valid_start="2016-01-11",
    valid_end="2020-01-01",
    test_start="2020-01-11",
    embargo=5,
)


def make_panel(periods: int = 3000) -> Panel:
    dates = pd.date_range("2010-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.02, (periods, 8)), axis=0),
        index=dates,
        columns=[f"S{i}" for i in range(8)],
    )
    return Panel.from_prices(
        open=close.shift(1).fillna(close.iloc[0]),
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=pd.DataFrame(rng.uniform(1e6, 5e6, close.shape), index=dates, columns=close.columns),
    )


def make_context(**overrides: object) -> AgentContext:
    guard = SplitGuard(BOUNDARIES)
    target = Node("rank", (Node("close"),))
    defaults: dict[str, object] = {
        "session_id": "s1",
        "session_name": "Session",
        "universe": "sp500-lite",
        "boundaries": BOUNDARIES,
        "guard": guard,
        "panel": truncate_panel(make_panel(), guard),
        "config": {"horizon": 1, "min_names": 5, "max_nodes": 40},
        "target_tree": target,
        "target_expanded": target,
    }
    defaults.update(overrides)
    return AgentContext(**defaults)  # type: ignore[arg-type]


def window(value: int) -> Node:
    return Node("window", value=value)


VOL = Node("rank", (Node("ts_std", (Node("returns"), window(20))),))


# --- the split itself ------------------------------------------------------------
def test_inner_split_is_chronological_and_embargoed() -> None:
    dates = pd.date_range("2010-01-01", periods=1000, freq="B")
    split = inner_split(dates, horizon=1, embargo=5)
    assert split.train.max() < split.holdout.min()
    gap = dates.get_loc(split.holdout.min()) - dates.get_loc(split.train.max())
    assert gap > split.embargo >= 5


def test_embargo_is_never_narrower_than_the_label_window() -> None:
    """A forward-return label on the last inner-train date must not reach the holdout."""
    dates = pd.date_range("2010-01-01", periods=2000, freq="B")
    split = inner_split(dates, horizon=20, embargo=5)
    assert split.embargo >= 20


def test_inner_split_refuses_a_window_too_short_to_be_meaningful() -> None:
    dates = pd.date_range("2010-01-01", periods=MIN_INNER_DATES, freq="B")
    with pytest.raises(ValueError, match="too short"):
        inner_split(dates, horizon=1, embargo=5)


def test_inner_split_reports_its_own_geometry() -> None:
    described = inner_split(
        pd.date_range("2010-01-01", periods=1000, freq="B"), horizon=1, embargo=5
    ).to_dict()
    assert described["inner_train_dates"] > 0 and described["inner_holdout_dates"] > 0
    assert described["inner_train_end"] < described["inner_holdout_start"]


# --- the invariant ---------------------------------------------------------------
def test_scoring_never_reads_a_date_past_the_training_boundary() -> None:
    context = make_context()
    score = score_candidate(VOL, context, expression="rank(ts_std(returns, 20))")
    assert pd.Timestamp(score.split["inner_holdout_end"]) <= pd.Timestamp(BOUNDARIES.train_end)
    assert pd.Timestamp(score.split["inner_holdout_end"]) < pd.Timestamp(BOUNDARIES.valid_start)


def test_scoring_fails_loudly_if_the_panel_was_widened_behind_the_guard() -> None:
    """Defence in depth: if a refactor ever hands the agent an untruncated panel, stop."""
    guard = SplitGuard(BOUNDARIES)
    leaky = make_context(panel=make_panel())  # deliberately NOT truncated
    assert pd.DatetimeIndex(leaky.panel.dates).max() > guard.train_end
    with pytest.raises(GuardViolation, match="past the frozen training boundary"):
        score_candidate(VOL, leaky, expression="rank(ts_std(returns, 20))")


# --- the scores ------------------------------------------------------------------
def test_score_reports_both_sides_of_the_inner_split() -> None:
    score = score_candidate(VOL, make_context(), expression="vol")
    assert score.inner_train["oriented_ic"] is not None
    assert score.inner_holdout["oriented_ic"] is not None
    assert score.inner_train["valid_dates"] and score.inner_holdout["valid_dates"]


def test_generalization_gap_is_train_minus_holdout() -> None:
    score = score_candidate(VOL, make_context(), expression="vol")
    expected = score.inner_train["oriented_ic"] - score.inner_holdout["oriented_ic"]
    assert score.generalization_gap == pytest.approx(expected, abs=1e-6)


def test_payload_states_what_the_number_is_not() -> None:
    """The model must never mistake an inner-holdout IC for validation evidence."""
    payload = score_candidate(VOL, make_context(), expression="vol").to_dict()
    assert "not the session's validation IC" in payload["measured_on"]
    assert "inner_train" in payload and "inner_holdout" in payload


def test_scoring_is_deterministic() -> None:
    context = make_context()
    first = score_candidate(VOL, context, expression="vol").to_dict()
    second = score_candidate(VOL, context, expression="vol").to_dict()
    assert first == second


def test_a_constant_factor_is_reported_infeasible_rather_than_scored() -> None:
    constant = Node("mul_scalar", (Node("close"), Node("const", value=0.0)))
    score = score_candidate(constant, make_context(), expression="zero")
    assert not score.feasible
    assert any("cross-section" in issue or "panel" in issue for issue in score.issues)


def test_backtest_block_carries_costs_and_can_be_skipped() -> None:
    context = make_context()
    with_bt = score_candidate(VOL, context, expression="vol")
    assert with_bt.backtest["costs_bps"] == {"commission": 1.0, "slippage": 5.0}
    without = score_candidate(VOL, context, expression="vol", run_backtest=False)
    assert without.backtest == {}


def test_note_is_carried_through_for_the_transcript() -> None:
    score = score_candidate(VOL, make_context(), expression="vol", note="testing a vol leg")
    assert score.note == "testing a vol leg"


# --- correlation with the incumbent ----------------------------------------------
def test_a_candidate_identical_to_the_target_correlates_at_one() -> None:
    target = Node("rank", (Node("close"),))
    context = make_context(target_tree=target, target_expanded=target)
    assert correlation_with_target(target, context) == pytest.approx(1.0, abs=1e-6)


def test_correlation_is_computed_only_on_training_dates() -> None:
    context = make_context()
    assert correlation_with_target(VOL, context) is not None
    context.guard.assert_within(context.train_dates)


def test_correlation_is_none_without_a_target() -> None:
    context = make_context(target_tree=None, target_expanded=None)
    assert correlation_with_target(VOL, context) is None
