"""P11-T3 - the tool registry and the DSL text parser.

The registry is the agent's whole capability, so these tests are mostly about what the tools
refuse: unknown operators, disabled operators, protected config keys, and anything that would
spend an evaluation without the budget being charged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalineage.agent.context import AgentContext, truncate_panel
from alphalineage.agent.expressions import ExpressionError, parse_expression
from alphalineage.agent.guards import Budget, BudgetLedger, GuardViolation, SplitGuard
from alphalineage.agent.tools import (
    ANNOTATE,
    EVALUATE,
    PROPOSE,
    READ,
    TOOLS,
    ToolRuntime,
    all_tools,
    catalog,
)
from alphalineage.api.sessions import Boundaries
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node
from alphalineage.explain.anatomy import formula_text

BOUNDARIES = Boundaries(
    train_end="2016-01-01",
    valid_start="2016-01-11",
    valid_end="2020-01-01",
    test_start="2020-01-11",
    embargo=5,
)


def make_panel(periods: int = 2200) -> Panel:
    dates = pd.date_range("2010-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(3)
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


def make_runtime(config: dict | None = None, budget: Budget | None = None) -> ToolRuntime:
    guard = SplitGuard(BOUNDARIES)
    target = Node("rank", (Node("close"),))
    context = AgentContext(
        session_id="s1",
        session_name="Session",
        universe="sp500-lite",
        boundaries=BOUNDARIES,
        guard=guard,
        panel=truncate_panel(make_panel(), guard),
        config=config
        or {
            "horizon": 1,
            "min_names": 5,
            "max_nodes": 40,
            "population_size": 200,
            "generations": 20,
        },
        target_tree=target,
        target_expanded=target,
        target_label="incumbent",
        target_metrics={"validation": {"oriented_ic": 0.05}},
        budget=BudgetLedger(budget=budget or Budget()),
    )
    return ToolRuntime(context)


# --- the parser ------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "rank(ts_std(returns, 20))",
        "sub(close, ts_mean(close, 20))",
        "ts_corr(returns, volume, 20)",
        "signed_power(rank(close), 0.5)",
        "where(gt(close, ts_mean(close, 20)), rank(returns), rank(neg(returns)))",
        "close",
    ],
)
def test_valid_expressions_round_trip_through_the_renderer(text: str) -> None:
    assert formula_text(parse_expression(text)) == formula_text(parse_expression(text))


def test_numbers_are_typed_by_the_slot_they_fill() -> None:
    windowed = parse_expression("ts_mean(close, 20)")
    assert windowed.children[1].name == "window" and windowed.children[1].value == 20
    scalar = parse_expression("mul_scalar(close, 0.5)")
    assert scalar.children[1].name == "const" and scalar.children[1].value == 0.5


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "empty"),
        ("rank(", "ended unexpectedly"),
        ("nonsense(close)", "unknown operator"),
        ("42", "bare number"),
        ("close(3)", "takes no arguments"),
        ("ts_mean(close, 1.5)", "whole number"),
        ("ts_mean(close)", "argument"),
        ("gt(close, open)", "must produce a series"),
        ("rank(close) trailing", "trailing"),
        ("rank(close) @", "unexpected character"),
    ],
)
def test_malformed_expressions_are_rejected_with_a_usable_message(text: str, fragment: str) -> None:
    with pytest.raises(ExpressionError) as excinfo:
        parse_expression(text)
    assert fragment in excinfo.value.message


def test_unknown_operator_error_points_at_the_discovery_tool() -> None:
    with pytest.raises(ExpressionError, match="list_operators"):
        parse_expression("magic_alpha(close)")


def test_an_operator_outside_the_allowed_set_is_refused() -> None:
    with pytest.raises(ExpressionError, match="not enabled"):
        parse_expression("rank(close)", allowed={"close", "ts_mean"})


def test_absurdly_long_input_is_refused_before_parsing() -> None:
    with pytest.raises(ExpressionError, match="characters"):
        parse_expression("rank(" * 2000)


# --- registry shape --------------------------------------------------------------
def test_catalog_exposes_every_tool_with_its_safety_class() -> None:
    entries = catalog()
    assert {item["name"] for item in entries} == set(TOOLS)
    assert {item["safety"] for item in entries} == {READ, EVALUATE, PROPOSE, ANNOTATE}
    for item in entries:
        assert item["description"] and item["parameters"]["type"] == "object"


def test_only_one_tool_can_spend_evidence() -> None:
    spenders = [item["name"] for item in catalog(kinds=(EVALUATE,))]
    assert spenders == ["evaluate_expression"]


def test_no_tool_accepts_a_date_universe_or_split_argument() -> None:
    """The split clamp holds because there is no parameter through which to ask."""
    forbidden = {"start", "end", "date", "dates", "split", "universe", "window_start"}
    for item in catalog():
        assert not forbidden & set(item["parameters"].get("properties", {}))


def test_there_is_one_toolset_and_the_model_chooses() -> None:
    """P11: no per-mode split. The user should not have to classify their own question."""
    names = {tool.name for tool in all_tools()}
    assert names == set(TOOLS)
    assert {"evaluate_expression", "propose_config_patch", "annotate_factor"} <= names


# --- class R: free and harmless --------------------------------------------------
def test_list_operators_returns_signatures_and_can_filter() -> None:
    runtime = make_runtime()
    result = runtime.execute("list_operators", {}).result
    assert result["count"] > 0
    assert any("->" in item["signature"] for item in result["operators"])
    filtered = runtime.execute("list_operators", {"category": "cross_sectional"}).result
    assert {item["category"] for item in filtered["operators"]} == {"cross_sectional"}


def test_validate_expression_is_free_and_charges_no_evaluation() -> None:
    runtime = make_runtime()
    call = runtime.execute("validate_expression", {"expression": "rank(ts_std(returns, 20))"})
    assert call.ok and call.result["valid"] is True
    assert call.result["effective_lookback_bars"] == 19
    assert runtime.context.budget.evaluations == 0
    assert runtime.context.trials.count == 0


def test_a_bad_expression_returns_an_error_the_model_can_act_on() -> None:
    runtime = make_runtime()
    call = runtime.execute("validate_expression", {"expression": "ts_mean(close)"})
    assert not call.ok
    assert "argument" in call.result["error"]
    assert "position" in call.result


def test_get_current_factor_labels_every_metric_with_its_split() -> None:
    result = make_runtime().execute("get_current_factor", {}).result
    assert result["expression"] == "rank(close)"
    assert "measurement" in result  # computed shape, not a hand-written label list
    assert "in-sample" in result["metrics_note"]
    assert "NOT a holdout result" in result["metrics_note"]


def test_get_session_config_discloses_what_may_not_be_touched() -> None:
    result = make_runtime().execute("get_session_config", {}).result
    assert "seed" in result["protected_keys"]
    assert "population_size" in result["tunable_keys"]
    assert result["context"]["reach"]["readable_through"] == "2016-01-01"


def test_unknown_tool_is_reported_rather_than_crashing() -> None:
    call = make_runtime().execute("delete_everything", {})
    assert not call.ok
    assert "unknown tool" in call.result["error"]
    assert "evaluate_expression" in call.result["available"]


# --- class E: budgeted and counted -----------------------------------------------
def test_evaluation_charges_a_trial_and_reports_both_sides() -> None:
    runtime = make_runtime()
    call = runtime.execute("evaluate_expression", {"expression": "rank(ts_std(returns, 20))"})
    assert call.ok
    assert runtime.context.trials.count == 1
    assert runtime.context.budget.evaluations == 1
    assert call.result["inner_train"]["oriented_ic"] is not None
    assert call.result["inner_holdout"]["oriented_ic"] is not None
    assert call.result["trials_spent_so_far"] == 1


def test_evaluation_result_states_it_is_not_validation_evidence() -> None:
    call = make_runtime().execute("evaluate_expression", {"expression": "rank(close)"})
    assert "not the session's validation IC" in call.result["measured_on"]


def test_evaluation_reports_correlation_with_the_incumbent() -> None:
    """A variant that is 0.99 correlated with the original is the same bet, not an improvement."""
    runtime = make_runtime()
    call = runtime.execute("evaluate_expression", {"expression": "rank(close)"})
    assert call.result["correlation_with_current_factor"] == pytest.approx(1.0, abs=1e-6)


def test_evaluation_budget_stops_the_run_rather_than_being_swallowed() -> None:
    runtime = make_runtime(budget=Budget(max_tool_calls=10, max_evaluations=1))
    runtime.execute("evaluate_expression", {"expression": "rank(close)"})
    from alphalineage.agent.guards import BudgetExhausted

    with pytest.raises(BudgetExhausted):
        runtime.execute("evaluate_expression", {"expression": "rank(returns)"})


def test_every_result_tells_the_model_what_budget_is_left() -> None:
    call = make_runtime().execute("list_operators", {})
    assert set(call.result["budget_remaining"]) == {
        "tool_calls_left",
        "evaluations_left",
        "seconds_left",
    }


def test_an_invalid_expression_still_costs_its_evaluation_slot() -> None:
    """Otherwise the budget could be circumvented by spamming malformed candidates."""
    runtime = make_runtime()
    runtime.execute("evaluate_expression", {"expression": "not_an_operator(close)"})
    assert runtime.context.budget.evaluations == 1
    assert runtime.context.trials.count == 0  # nothing was actually scored


# --- class P: staged, inert ------------------------------------------------------
def test_proposing_a_factor_stages_it_without_saving() -> None:
    runtime = make_runtime()
    call = runtime.execute(
        "propose_factor",
        {
            "expression": "rank(ts_corr(returns, volume, 20))",
            "name": "volume leg",
            "rationale": "adds the volume aspect the incumbent lacks",
        },
    )
    assert call.ok and call.result["staged"] is True
    assert len(runtime.proposals) == 1
    assert runtime.proposals[0].applied is False
    assert "nothing has happened yet" in call.result["note"]


def test_a_proposal_without_a_rationale_is_refused() -> None:
    call = make_runtime().execute(
        "propose_factor", {"expression": "rank(close)", "name": "x", "rationale": "  "}
    )
    assert not call.ok and "rationale is required" in call.result["error"]


def test_config_patch_is_staged_with_a_reviewable_diff() -> None:
    runtime = make_runtime()
    call = runtime.execute(
        "propose_config_patch",
        {"patch": {"generations": 40}, "rationale": "fitness was still climbing at the end"},
    )
    assert call.ok
    assert call.result["diff"] == [{"key": "generations", "from": 20, "to": 40}]
    assert "unchanged" in call.result["note"]


def test_a_protected_config_key_is_refused_with_the_reason() -> None:
    call = make_runtime().execute(
        "propose_config_patch", {"patch": {"seed": 7}, "rationale": "try a luckier run"}
    )
    assert not call.ok
    assert "seed-hacking" in call.result["error"]
    assert call.result["rejected_patch"] == {}


def test_proposals_do_not_mutate_the_session_config() -> None:
    runtime = make_runtime()
    before = dict(runtime.context.config)
    runtime.execute(
        "propose_config_patch",
        {"patch": {"generations": 40}, "rationale": "more budget"},
    )
    assert runtime.context.config == before


# --- the invariant ---------------------------------------------------------------
def test_no_tool_can_read_past_the_training_boundary() -> None:
    """Drive every tool and assert the guard afterwards."""
    runtime = make_runtime()
    for name, args in (
        ("list_operators", {}),
        ("validate_expression", {"expression": "rank(close)"}),
        ("measure_expression", {"expression": "rank(close)"}),
        ("get_current_factor", {}),
        ("get_search_trajectory", {}),
        ("get_session_config", {}),
        ("evaluate_expression", {"expression": "rank(ts_std(returns, 20))"}),
        ("propose_factor", {"expression": "rank(close)", "name": "n", "rationale": "r"}),
        ("propose_config_patch", {"patch": {"generations": 30}, "rationale": "r"}),
    ):
        runtime.execute(name, args)
    runtime.context.guard.assert_within(runtime.context.panel.dates)
    assert pd.DatetimeIndex(runtime.context.panel.dates).max() <= pd.Timestamp(BOUNDARIES.train_end)


def test_guard_violations_are_not_swallowed_as_tool_errors() -> None:
    """A guard breach must fail the run, not come back as a retryable message."""
    guard = SplitGuard(BOUNDARIES)
    target = Node("rank", (Node("close"),))
    leaky = AgentContext(
        session_id="s1",
        session_name="Session",
        universe="u",
        boundaries=BOUNDARIES,
        guard=guard,
        panel=make_panel(),  # untruncated on purpose
        config={"horizon": 1, "min_names": 5},
        target_tree=target,
        target_expanded=target,
    )
    with pytest.raises(GuardViolation):
        ToolRuntime(leaky).execute("evaluate_expression", {"expression": "rank(close)"})


# --- P11 additions ----------------------------------------------------------------
def test_measure_expression_returns_computed_numbers_not_labels() -> None:
    result = (
        make_runtime()
        .execute("measure_expression", {"expression": "rank(ts_std(returns, 20))"})
        .result
    )
    assert "Lookback" in result["measurement"]
    assert "compounded" in result["note"]
    assert "aspects" not in result  # the hand-written taxonomy is gone


def test_annotate_factor_attaches_labels_without_needing_approval() -> None:
    runtime = make_runtime()
    call = runtime.execute(
        "annotate_factor",
        {
            "labels": [
                {
                    "name": "dispersion of recent returns",
                    "evidence": "ts_std(returns, 20)",
                    "reading": "size of variation, not direction",
                }
            ]
        },
    )
    assert call.ok and call.result["attached"] == 1
    assert runtime.labels[0]["name"] == "dispersion of recent returns"
    assert runtime.proposals == []  # an annotation is not a proposal


def test_annotate_factor_accepts_a_vocabulary_no_fixed_list_could_hold() -> None:
    """The point of deleting the taxonomy: the model names what it actually sees."""
    runtime = make_runtime()
    runtime.execute(
        "annotate_factor",
        {"labels": [{"name": "overnight-gap sensitivity"}, {"name": "earnings-window blackout"}]},
    )
    assert [item["name"] for item in runtime.labels] == [
        "overnight-gap sensitivity",
        "earnings-window blackout",
    ]


def test_annotate_factor_rejects_an_empty_or_nameless_list() -> None:
    runtime = make_runtime()
    assert not runtime.execute("annotate_factor", {"labels": []}).ok
    assert not runtime.execute("annotate_factor", {"labels": [{"evidence": "x"}]}).ok


def test_search_workspace_requires_a_query() -> None:
    assert not make_runtime().execute("search_workspace", {}).ok


def test_list_agent_rounds_is_empty_without_a_session(temp_data_dir) -> None:
    result = make_runtime().execute("list_agent_rounds", {}).result
    assert result["count"] == 0
