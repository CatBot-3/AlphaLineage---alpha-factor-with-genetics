"""P11-T4 - the conversational LangGraph loop and the provider runtime.

The whole loop is driven by an injected model, so every path here runs with no network, no API
key, and no tokens: a scripted fake returns the tool calls a real model would emit. That is what
makes the awkward cases — budget exhaustion mid-turn, a malformed tool call, a provider outage —
testable at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from alphalineage.agent.context import AgentContext, truncate_panel
from alphalineage.agent.conversation import Conversation
from alphalineage.agent.graph import (
    SYSTEM_PROMPT,
    TurnResult,
    run_turn,
    session_header,
)
from alphalineage.agent.guards import Budget, BudgetLedger, SplitGuard
from alphalineage.agent.runtime import AgentUnavailable, build_chat_model, supports_tool_calling
from alphalineage.api.sessions import Boundaries
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node
from alphalineage.explain.credentials import LLMConfig, ResolvedKey
from alphalineage.explain.providers import LLMError

BOUNDARIES = Boundaries(
    train_end="2016-01-01",
    valid_start="2016-01-11",
    valid_end="2020-01-01",
    test_start="2020-01-11",
    embargo=5,
)
KEY = "sk-agent-secret-key-value-0123456789"


def make_panel(periods: int = 2200) -> Panel:
    dates = pd.date_range("2010-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(5)
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


def make_context(budget: Budget | None = None) -> AgentContext:
    guard = SplitGuard(BOUNDARIES)
    target = Node("rank", (Node("close"),))
    return AgentContext(
        session_id="s1",
        session_name="Volatility hunt",
        universe="sp500-lite",
        boundaries=BOUNDARIES,
        guard=guard,
        panel=truncate_panel(make_panel(), guard),
        config={
            "horizon": 1,
            "min_names": 5,
            "max_nodes": 40,
            "population_size": 200,
            "generations": 20,
        },
        target_tree=target,
        target_expanded=target,
        target_label="incumbent",
        budget=BudgetLedger(budget=budget or Budget()),
    )


class ScriptedModel:
    """A stand-in for a bound chat model: replays a fixed sequence of turns."""

    def __init__(self, *turns: AIMessage) -> None:
        self.turns = list(turns)
        self.calls = 0
        self.bound_tools: list[dict[str, Any]] = []
        self.seen_messages: list[list[Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]) -> ScriptedModel:
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.seen_messages.append(list(messages))
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn


def ai(text: str = "", tool_calls: list[dict[str, Any]] | None = None) -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": c["name"], "args": c.get("args", {}), "id": c.get("id", f"call{i}")}
            for i, c in enumerate(tool_calls or [])
        ],
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )


# --- the prompt and the header -----------------------------------------------------
def test_one_prompt_carries_the_honesty_rules_for_every_kind_of_request() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())  # the rules are hard-wrapped in the source
    assert "Use ONLY operators returned by list_operators" in prompt
    assert "cannot read the validation window or the locked test split" in prompt
    assert "rank IC, not return" in prompt
    assert "not investment advice" in prompt
    assert "You are proposing, not deciding" in prompt


def test_the_prompt_tells_the_model_to_decide_which_tools_a_request_needs() -> None:
    """P11: one door. The model classifies the question, not the user."""
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "Decide for yourself which tools a request needs" in prompt
    assert "Asked what a factor measures" in prompt
    assert "Asked to improve a factor" in prompt


def test_the_prompt_forbids_estimating_lookback_by_eye() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "Do not work out a factor's lookback by eye" in prompt
    assert "measure_expression" in prompt


def test_the_header_states_the_reach_before_any_work() -> None:
    text = session_header(make_context())
    assert "training data through 2016-01-01" in text
    assert "neither is reachable from any tool you have" in text
    assert "rank(close)" in text


# --- the loop ----------------------------------------------------------------------
def test_a_single_prose_turn_completes_immediately() -> None:
    result = run_turn(
        Conversation(session_id="s1"),
        make_context(),
        message="improve this",
        model=ScriptedModel(ai("Nothing to add.")),
    )
    assert result.stop_reason == "completed"
    assert result.text == "Nothing to add."
    assert result.calls == []


def test_tool_calls_are_executed_and_fed_back() -> None:
    model = ScriptedModel(
        ai(tool_calls=[{"name": "get_current_factor"}]),
        ai(tool_calls=[{"name": "validate_expression", "args": {"expression": "rank(returns)"}}]),
        ai("I looked at the incumbent and checked one variant."),
    )
    result = run_turn(
        Conversation(session_id="s1"), make_context(), message="improve this", model=model
    )
    assert result.stop_reason == "completed"
    assert [call.tool for call in result.calls] == [
        "get_current_factor",
        "validate_expression",
    ]
    assert all(call.ok for call in result.calls)
    assert model.calls == 3


def test_every_tool_is_bound_because_the_model_chooses() -> None:
    model = ScriptedModel(ai("done"))
    run_turn(Conversation(session_id="s1"), make_context(), message="tune it", model=model)
    names = {tool["function"]["name"] for tool in model.bound_tools}
    assert {"propose_config_patch", "evaluate_expression", "annotate_factor"} <= names


def test_bound_tool_schema_is_the_openai_function_shape() -> None:
    model = ScriptedModel(ai("done"))
    run_turn(Conversation(session_id="s1"), make_context(), message="improve", model=model)
    tool = model.bound_tools[0]
    assert tool["type"] == "function"
    assert set(tool["function"]) == {"name", "description", "parameters"}
    assert tool["function"]["parameters"]["type"] == "object"


def test_an_evaluation_is_counted_as_a_trial_through_the_loop() -> None:
    context = make_context()
    model = ScriptedModel(
        ai(
            tool_calls=[
                {"name": "evaluate_expression", "args": {"expression": "rank(ts_std(returns, 20))"}}
            ]
        ),
        ai("That variant held up on the inner holdout."),
    )
    result = run_turn(Conversation(session_id="s1"), context, message="improve this", model=model)
    assert result.usage["trials"] == 1
    assert result.usage["evaluations"] == 1


def test_staged_proposals_come_back_unapproved() -> None:
    model = ScriptedModel(
        ai(
            tool_calls=[
                {
                    "name": "propose_factor",
                    "args": {
                        "expression": "rank(ts_corr(returns, volume, 20))",
                        "name": "volume leg",
                        "rationale": "adds the volume aspect the incumbent lacks",
                    },
                }
            ]
        ),
        ai("Staged one candidate."),
    )
    result = run_turn(
        Conversation(session_id="s1"), make_context(), message="improve this", model=model
    )
    assert len(result.proposals) == 1
    assert result.proposals[0]["applied"] is False
    assert result.proposals[0]["kind"] == "factor"


def test_token_usage_is_charged_from_the_provider_metadata() -> None:
    context = make_context()
    run_turn(Conversation(session_id="s1"), context, message="hi", model=ScriptedModel(ai("done")))
    assert context.budget.input_tokens == 100
    assert context.budget.output_tokens == 20


# --- budget behaviour ---------------------------------------------------------------
def test_budget_exhaustion_gives_the_model_a_final_turn_to_summarize() -> None:
    """A stopped agent that never got to say what it found is a wasted spend."""
    context = make_context(budget=Budget(max_tool_calls=10, max_evaluations=1))
    model = ScriptedModel(
        ai(tool_calls=[{"name": "evaluate_expression", "args": {"expression": "rank(close)"}}]),
        ai("Budget spent. The one candidate I tested did not beat the incumbent."),
    )
    result = run_turn(Conversation(session_id="s1"), context, message="improve this", model=model)
    assert result.stop_reason == "budget_exhausted"
    assert "did not beat the incumbent" in result.text
    assert result.usage["exhausted_reason"]


def test_the_wrap_up_turn_tells_the_model_to_stop_calling_tools() -> None:
    context = make_context(budget=Budget(max_tool_calls=10, max_evaluations=1))
    model = ScriptedModel(
        ai(tool_calls=[{"name": "evaluate_expression", "args": {"expression": "rank(close)"}}]),
        ai("Summary."),
    )
    run_turn(Conversation(session_id="s1"), context, message="improve this", model=model)
    last_turn = model.seen_messages[-1]
    assert any("Do not call any more tools" in getattr(m, "content", "") for m in last_turn)


def test_ignoring_the_wrap_up_and_calling_tools_again_ends_the_run() -> None:
    context = make_context(budget=Budget(max_tool_calls=10, max_evaluations=1))
    model = ScriptedModel(
        ai(tool_calls=[{"name": "evaluate_expression", "args": {"expression": "rank(close)"}}]),
        ai(tool_calls=[{"name": "evaluate_expression", "args": {"expression": "rank(open)"}}]),
        ai("never reached"),
    )
    result = run_turn(Conversation(session_id="s1"), context, message="improve this", model=model)
    assert result.stop_reason == "budget_exhausted"
    assert context.budget.evaluations == 1  # the second evaluation never ran


def test_the_tool_call_budget_also_stops_the_loop() -> None:
    context = make_context(budget=Budget(max_tool_calls=2, max_evaluations=5))
    model = ScriptedModel(
        ai(tool_calls=[{"name": "list_operators"}]),
        ai(tool_calls=[{"name": "list_operators"}]),
        ai("Out of calls."),
    )
    result = run_turn(Conversation(session_id="s1"), context, message="improve this", model=model)
    assert result.stop_reason == "budget_exhausted"
    assert context.budget.tool_calls == 2


# --- failure handling ---------------------------------------------------------------
def test_a_malformed_tool_call_is_returned_to_the_model_not_fatal() -> None:
    model = ScriptedModel(
        ai(tool_calls=[{"name": "validate_expression", "args": {"expression": "ts_mean(close)"}}]),
        ai("That did not typecheck; I would need a window argument."),
    )
    result = run_turn(
        Conversation(session_id="s1"), make_context(), message="improve this", model=model
    )
    assert result.stop_reason == "completed"
    assert result.calls[0].ok is False
    assert "argument" in result.calls[0].result["error"]


def test_an_unknown_tool_name_is_reported_and_the_run_continues() -> None:
    model = ScriptedModel(
        ai(tool_calls=[{"name": "rm_rf_slash", "args": {}}]),
        ai("That tool does not exist."),
    )
    result = run_turn(
        Conversation(session_id="s1"), make_context(), message="improve this", model=model
    )
    assert result.stop_reason == "completed"
    assert "unknown tool" in result.calls[0].result["error"]


def test_a_provider_outage_is_a_result_not_an_exception() -> None:
    class Broken:
        def bind_tools(self, tools: list[dict[str, Any]]) -> Broken:
            return self

        def invoke(self, messages: list[Any]) -> AIMessage:
            raise RuntimeError("connection reset")

    result = run_turn(
        Conversation(session_id="s1"), make_context(), message="improve this", model=Broken()
    )
    assert result.stop_reason == "failed"
    assert "connection reset" in result.error


def test_result_converts_into_a_conversation_turn() -> None:
    payload = (
        run_turn(
            Conversation(session_id="s1"),
            make_context(),
            message="improve this",
            model=ScriptedModel(ai("done")),
        )
        .to_turn()
        .to_dict()
    )
    assert set(payload) >= {
        "role",
        "text",
        "calls",
        "labels",
        "proposals",
        "usage",
        "stop_reason",
    }
    assert payload["role"] == "assistant"
    assert isinstance(TurnResult().to_turn().to_dict(), dict)


# --- the invariant -------------------------------------------------------------------
def test_a_full_loop_never_reads_past_the_training_boundary() -> None:
    context = make_context()
    model = ScriptedModel(
        ai(tool_calls=[{"name": "get_current_factor"}]),
        ai(
            tool_calls=[
                {"name": "evaluate_expression", "args": {"expression": "rank(ts_std(returns, 20))"}}
            ]
        ),
        ai(
            tool_calls=[
                {
                    "name": "propose_factor",
                    "args": {
                        "expression": "rank(ts_std(returns, 20))",
                        "name": "vol",
                        "rationale": "held up on the inner holdout",
                    },
                }
            ]
        ),
        ai("Done."),
    )
    run_turn(Conversation(session_id="s1"), context, message="improve this", model=model)
    context.guard.assert_within(context.panel.dates)
    assert pd.DatetimeIndex(context.panel.dates).max() <= pd.Timestamp(BOUNDARIES.train_end)


# --- the runtime and its egress guard ------------------------------------------------
def _config(provider: str, model: str, base_url: str = "") -> LLMConfig:
    return LLMConfig(
        provider=provider, model=model, base_url=base_url, key=ResolvedKey(KEY, "stored")
    )


def test_a_named_provider_cannot_be_pointed_at_another_host() -> None:
    """LangChain does the HTTP, so the allowlist is re-asserted before a client is built."""
    with pytest.raises(LLMError, match="may only call"):
        build_chat_model(_config("openai", "gpt-4o", "https://evil.example.com/v1"))


def test_a_custom_endpoint_must_still_be_https_or_loopback() -> None:
    with pytest.raises(LLMError, match="must use https"):
        build_chat_model(_config("openai_compatible", "local", "http://example.com/v1"))


def test_a_missing_key_or_model_is_refused_before_a_client_exists() -> None:
    with pytest.raises(LLMError, match="no API key"):
        build_chat_model(
            LLMConfig(provider="openai", model="gpt-4o", base_url="", key=ResolvedKey("", "none"))
        )
    with pytest.raises(LLMError, match="no model"):
        build_chat_model(_config("openai", ""))


def test_building_a_supported_provider_succeeds_without_a_network_call() -> None:
    model = build_chat_model(_config("openai", "gpt-4o"))
    assert hasattr(model, "bind_tools")
    assert hasattr(build_chat_model(_config("anthropic", "claude-sonnet-4-5")), "bind_tools")
    assert hasattr(build_chat_model(_config("deepseek", "deepseek-chat")), "bind_tools")


def test_a_reasoning_only_model_is_refused_up_front_with_the_reason() -> None:
    ok, reason = supports_tool_calling(_config("deepseek", "deepseek-reasoner"))
    assert not ok and "function calling" in reason
    assert supports_tool_calling(_config("deepseek", "deepseek-chat"))[0]


def test_agent_unavailable_is_a_distinct_error_type() -> None:
    assert issubclass(AgentUnavailable, RuntimeError)
