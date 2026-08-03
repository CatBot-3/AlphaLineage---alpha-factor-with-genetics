"""P10-T1 - the agent guard layer.

These are the load-bearing tests of the agent phase. The whole feature is only safe because an
agent cannot reach the locked splits, cannot spend unbounded evidence, and cannot move a
parameter that changes what a metric means. If any test here regresses, the agent is unsafe, not
merely broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalineage.agent.context import MIN_TRAIN_DATES, AgentContext, truncate_panel
from alphalineage.agent.guards import (
    DEFAULT_MAX_EVALUATIONS,
    MAX_EVALUATIONS_CEILING,
    PROTECTED_KEYS,
    TUNABLE_KEYS,
    Budget,
    BudgetExhausted,
    BudgetLedger,
    GuardViolation,
    SplitGuard,
    TrialLedger,
    validate_config_patch,
)
from alphalineage.api.sessions import Boundaries
from alphalineage.core.panel import Panel

BOUNDARIES = Boundaries(
    train_end="2016-01-01",
    valid_start="2016-01-11",
    valid_end="2020-01-01",
    test_start="2020-01-11",
    embargo=5,
)


def make_panel(start: str = "2010-01-01", periods: int = 3000) -> Panel:
    dates = pd.date_range(start, periods=periods, freq="B")
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.02, (periods, 6)), axis=0),
        index=dates,
        columns=[f"S{i}" for i in range(6)],
    )
    return Panel.from_prices(
        open=close.shift(1).fillna(close.iloc[0]),
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=pd.DataFrame(rng.uniform(1e6, 5e6, close.shape), index=dates, columns=close.columns),
    )


# --- the split guard -------------------------------------------------------------
def test_allowed_stops_at_the_training_boundary() -> None:
    guard = SplitGuard(BOUNDARIES)
    dates = pd.date_range("2015-12-20", periods=40, freq="B")
    allowed = guard.allowed(dates)
    assert allowed.max() <= pd.Timestamp("2016-01-01")
    assert allowed.max() < pd.Timestamp(BOUNDARIES.valid_start)


def test_assert_within_rejects_a_validation_date() -> None:
    guard = SplitGuard(BOUNDARIES)
    with pytest.raises(GuardViolation, match="past the frozen training boundary"):
        guard.assert_within(pd.date_range("2016-01-11", periods=5, freq="B"))


def test_assert_within_rejects_a_test_date() -> None:
    guard = SplitGuard(BOUNDARIES)
    with pytest.raises(GuardViolation):
        guard.assert_within(pd.DatetimeIndex([pd.Timestamp(BOUNDARIES.test_start)]))


def test_assert_within_accepts_an_empty_index() -> None:
    SplitGuard(BOUNDARIES).assert_within(pd.DatetimeIndex([]))


def test_inconsistent_boundaries_are_rejected_at_construction() -> None:
    broken = Boundaries(
        train_end="2016-01-01",
        valid_start="2015-06-01",  # before train_end
        valid_end="2020-01-01",
        test_start="2020-01-11",
        embargo=5,
    )
    with pytest.raises(GuardViolation, match="inconsistent"):
        SplitGuard(broken)


def test_truncation_removes_validation_and_test_rows_from_memory() -> None:
    """The clamp is enforced by absence, not by a conditional: the rows are simply gone."""
    guard = SplitGuard(BOUNDARIES)
    panel = truncate_panel(make_panel(), guard)
    dates = pd.DatetimeIndex(panel.dates)
    assert dates.max() <= pd.Timestamp(BOUNDARIES.train_end)
    assert not (dates >= pd.Timestamp(BOUNDARIES.valid_start)).any()
    assert not (dates >= pd.Timestamp(BOUNDARIES.test_start)).any()
    for frame in panel.fields.values():
        assert frame.index.max() <= pd.Timestamp(BOUNDARIES.train_end)


def test_truncation_refuses_a_panel_with_no_training_dates() -> None:
    guard = SplitGuard(BOUNDARIES)
    late = make_panel(start="2021-01-01", periods=400)
    with pytest.raises(GuardViolation, match="no cached dates"):
        truncate_panel(late, guard)


@pytest.mark.parametrize(
    "probe",
    [
        "2016-01-04",  # first business day past train_end
        "2016-01-11",  # valid_start
        "2018-06-15",  # mid validation
        "2020-01-11",  # test_start
        "2026-01-01",  # far future
    ],
)
def test_no_probe_date_past_the_boundary_survives_truncation(probe: str) -> None:
    """Fuzz the boundary: whatever date a caller hopes to reach, it is not in the panel."""
    guard = SplitGuard(BOUNDARIES)
    panel = truncate_panel(make_panel(), guard)
    assert pd.Timestamp(probe) not in pd.DatetimeIndex(panel.dates)


def test_describe_states_what_is_and_is_not_readable() -> None:
    described = SplitGuard(BOUNDARIES).describe()
    assert described["readable_through"] == "2016-01-01"
    assert described["validation_starts"] == "2016-01-11"
    assert described["test_starts"] == "2020-01-11"
    assert "training-window data only" in described["note"]


# --- the context -----------------------------------------------------------------
def _context(**overrides: object) -> AgentContext:
    guard = SplitGuard(BOUNDARIES)
    defaults: dict[str, object] = {
        "session_id": "s1",
        "session_name": "Session",
        "universe": "sp500-lite",
        "boundaries": BOUNDARIES,
        "guard": guard,
        "panel": truncate_panel(make_panel(), guard),
        "config": {"horizon": 1, "population_size": 200, "generations": 20},
    }
    defaults.update(overrides)
    return AgentContext(**defaults)  # type: ignore[arg-type]


def test_context_exposes_only_training_dates() -> None:
    context = _context()
    context.guard.assert_within(context.train_dates)
    assert context.train_dates.max() <= pd.Timestamp(BOUNDARIES.train_end)


def test_context_refuses_a_training_window_too_short_to_split() -> None:
    guard = SplitGuard(BOUNDARIES)
    short = Panel(
        {
            name: frame.iloc[: MIN_TRAIN_DATES - 1]
            for name, frame in truncate_panel(make_panel(), guard).fields.items()
        }
    )
    with pytest.raises(GuardViolation, match="at least"):
        _context(panel=short)


def test_context_describe_carries_the_reach_statement() -> None:
    described = _context().describe()
    assert described["reach"]["readable_through"] == "2016-01-01"
    assert described["train_dates"] > 0


# --- budgets ---------------------------------------------------------------------
def test_budget_defaults_are_conservative() -> None:
    budget = Budget()
    assert budget.max_evaluations == DEFAULT_MAX_EVALUATIONS
    assert budget.max_evaluations < MAX_EVALUATIONS_CEILING


def test_budget_rejects_values_above_the_ceiling() -> None:
    with pytest.raises(ValueError, match="at most"):
        Budget(max_evaluations=MAX_EVALUATIONS_CEILING + 1)
    with pytest.raises(ValueError, match="positive integer"):
        Budget(max_tool_calls=0)
    with pytest.raises(ValueError, match="max_seconds"):
        Budget(max_seconds=0)


def test_tool_call_budget_stops_the_loop() -> None:
    ledger = BudgetLedger(budget=Budget(max_tool_calls=2))
    ledger.charge_tool_call()
    ledger.charge_tool_call()
    assert ledger.exhausted
    with pytest.raises(BudgetExhausted, match="tool-call budget"):
        ledger.charge_tool_call()


def test_evaluation_budget_is_separate_from_tool_calls() -> None:
    ledger = BudgetLedger(budget=Budget(max_tool_calls=10, max_evaluations=1))
    ledger.charge_evaluation()
    with pytest.raises(BudgetExhausted, match="evaluation budget"):
        ledger.charge_evaluation()
    ledger.charge_tool_call()  # free tools still work after evaluations run out


def test_wall_clock_budget_uses_the_injected_clock() -> None:
    now = [0.0]
    ledger = BudgetLedger(budget=Budget(max_seconds=10), clock=lambda: now[0])
    assert not ledger.exhausted
    now[0] = 11.0
    assert ledger.exhausted
    assert "time budget" in (ledger.exhausted_reason() or "")


def test_token_budget_is_tracked_across_both_directions() -> None:
    ledger = BudgetLedger(budget=Budget(max_tokens=100))
    ledger.charge_tokens(input_tokens=60, output_tokens=30)
    assert not ledger.exhausted
    ledger.charge_tokens(output_tokens=20)
    assert ledger.total_tokens == 110
    assert "token budget" in (ledger.exhausted_reason() or "")


def test_remaining_tells_the_model_what_is_left() -> None:
    ledger = BudgetLedger(budget=Budget(max_tool_calls=5, max_evaluations=3))
    ledger.charge_tool_call()
    ledger.charge_evaluation()
    remaining = ledger.remaining()
    assert remaining["tool_calls_left"] == 4
    assert remaining["evaluations_left"] == 2


# --- trials ----------------------------------------------------------------------
def test_every_evaluation_is_a_counted_trial() -> None:
    trials = TrialLedger()
    trials.record("rank(close)")
    trials.record("rank(ts_std(returns, 20))")
    assert trials.count == 2 and trials.distinct == 2


def test_re_evaluating_the_same_expression_still_costs_a_trial() -> None:
    """Distinct expressions matter for the search space; every look still costs a trial."""
    trials = TrialLedger()
    trials.record("rank(close)")
    trials.record("rank(close)")
    assert trials.count == 2
    assert trials.distinct == 1


# --- config patches --------------------------------------------------------------
BASE_CONFIG = {
    "population_size": 200,
    "generations": 20,
    "crossover_rate": 0.8,
    "point_mutation_rate": 0.1,
    "max_depth": 6,
    "max_nodes": 40,
    "horizon": 1,
    "ic_method": "spearman",
    "min_names": 5,
    "seed": 0,
}


def test_a_valid_patch_produces_a_reviewable_diff() -> None:
    result = validate_config_patch({"generations": 40, "population_size": 300}, BASE_CONFIG)
    assert result.ok
    assert {item["key"] for item in result.diff} == {"generations", "population_size"}
    assert result.merged["generations"] == 40
    assert result.merged["horizon"] == 1  # untouched keys survive


def test_protected_keys_are_refused_with_the_reason() -> None:
    for key, value in (("horizon", 5), ("ic_method", "pearson"), ("min_names", 1), ("seed", 42)):
        result = validate_config_patch({key: value}, BASE_CONFIG)
        assert not result.ok
        assert any(key in message for message in result.errors)
        assert PROTECTED_KEYS[key].split()[0] in " ".join(result.errors)


def test_seed_hacking_is_named_as_such() -> None:
    result = validate_config_patch({"seed": 99}, BASE_CONFIG)
    assert not result.ok
    assert "seed-hacking" in " ".join(result.errors)


def test_unknown_keys_are_refused_and_the_allowlist_is_disclosed() -> None:
    result = validate_config_patch({"nonsense": 1}, BASE_CONFIG)
    assert not result.ok
    assert "not a tunable search parameter" in result.errors[0]
    assert "population_size" in result.errors[0]


def test_out_of_bounds_values_come_back_as_retryable_errors_not_exceptions() -> None:
    result = validate_config_patch({"population_size": -5}, BASE_CONFIG)
    assert not result.ok and result.errors
    assert "population_size" in result.errors[0]


def test_the_evaluation_ceiling_is_enforced_against_the_planned_generations() -> None:
    result = validate_config_patch({"population_size": 5000}, BASE_CONFIG, generations=1000)
    assert not result.ok
    assert "evaluation ceiling" in result.errors[0]


def test_a_no_op_patch_is_rejected_so_the_user_is_not_asked_to_approve_nothing() -> None:
    result = validate_config_patch({"generations": 20}, BASE_CONFIG)
    assert not result.ok
    assert "does not change anything" in result.errors[0]


def test_an_empty_or_malformed_patch_is_rejected() -> None:
    assert not validate_config_patch({}, BASE_CONFIG).ok
    assert not validate_config_patch([], BASE_CONFIG).ok  # type: ignore[arg-type]


def test_tunable_and_protected_key_sets_do_not_overlap() -> None:
    assert not TUNABLE_KEYS & set(PROTECTED_KEYS)
