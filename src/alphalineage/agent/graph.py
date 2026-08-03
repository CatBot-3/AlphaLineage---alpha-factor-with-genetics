"""P11-T4 - one conversational turn, as a LangGraph state machine.

Two nodes and one conditional edge: the model proposes tool calls, the runtime executes them
under budget, and results go back as tool messages until the model answers without asking for
anything else. That answer plus its tool calls become one assistant turn in the conversation.

The change from Phase 10 is what drives the loop. There is no longer an "improve factor" mode and
an "explain factor" mode with different toolsets — the user writes a message, the model gets every
tool, and it decides. "What does this measure" reaches for the read tools and `annotate_factor`;
"make it better" reaches for `evaluate_expression` and `propose_factor`. The guards do not change
with the phrasing, so neither does the blast radius.

When a budget runs out the loop does not cut the model off: it injects a wrap-up message and gives
it one final turn to write its conclusion. A stopped agent that never got to say what it found is
a wasted spend.

Driven by an injected model object. Anything exposing ``bind_tools(...) -> .invoke`` works, which
is what lets the whole loop be tested without a network or an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphalineage.agent.context import AgentContext
from alphalineage.agent.conversation import Conversation, ToolCallRecord, Turn
from alphalineage.agent.guards import BudgetExhausted
from alphalineage.agent.memory import build_messages
from alphalineage.agent.tools import Tool, ToolRuntime, all_tools
from alphalineage.explain.anatomy import formula_text

MAX_STEPS = 24  # a backstop under the budgets, so a pathological model cannot spin forever

_WRAP_UP = (
    "Your budget for this turn is spent ({reason}). Do not call any more tools. Answer now with "
    "what you found, what the numbers actually showed, and what you would do next."
)

SYSTEM_PROMPT = """\
You are a quantitative research assistant inside AlphaLineage, an evolutionary alpha-factor
mining platform. You work through tools, in conversation with one researcher, on one training
session.

Factors are strongly-typed expression trees written in function-call form, for example
rank(ts_std(returns, 20)).

Decide for yourself which tools a request needs. Some shapes:
- Asked what a factor measures or how it works: get_current_factor, then measure_expression if
  you need more detail, then annotate_factor to attach your labels so they render beside your
  answer. There is no fixed label vocabulary — name what you actually see.
- Asked to improve a factor: check list_agent_rounds first so you build on your own earlier work,
  form a specific hypothesis, validate_expression (free) to get it compiling, evaluate_expression
  on the few you believe in, then propose_factor for what is genuinely worth a human's time.
- Asked why the search behaved as it did, or how to tune it: get_search_trajectory and
  get_session_config, then propose_config_patch with one well-argued change.
- Needing background you were not given: search_workspace.

Rules you must follow:
1. Use ONLY operators returned by list_operators. Operators cannot be invented, and you cannot
   add one to the system. If an idea is not expressible with the listed operators, say so.
2. validate_expression is free and unbudgeted. Use it to fix syntax and typing BEFORE spending an
   evaluation. Evaluations are strictly limited and each one costs a trial.
3. Every number you are given is labelled with the split it came from. Training and inner-holdout
   figures are in-sample research signals, not evidence of performance. Never describe them as
   out-of-sample results, and never imply a factor has been validated.
4. evaluate_expression measures on an inner holdout carved out of the TRAINING window. It cannot
   read the validation window or the locked test split, and neither can you. Do not claim
   otherwise, and do not ask for it.
5. Do not work out a factor's lookback by eye. Nested windows compound and the total is much
   larger than the biggest window; call measure_expression and use the number it returns.
6. Fitness here is rank IC, not return. Do not convert an IC into a return or a Sharpe claim.
7. You are proposing, not deciding. Everything you stage is reviewed by a human before it takes
   effect. Say plainly when you are unsure or when the evidence is thin. Staging nothing is a
   legitimate outcome; say so rather than padding a list.
8. This is research tooling, not investment advice.

Write like a colleague: direct, specific, no preamble. Cite the numbers you actually saw."""


@dataclass
class TurnResult:
    """The outcome of one exchange, ready to be appended to the conversation."""

    text: str = ""
    stop_reason: str = "completed"
    calls: list[ToolCallRecord] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_turn(self) -> Turn:
        from alphalineage.agent.conversation import ASSISTANT

        return Turn(
            role=ASSISTANT,
            text=self.text,
            calls=self.calls,
            labels=self.labels,
            proposals=self.proposals,
            usage=self.usage,
            stop_reason=self.stop_reason,
            error=self.error,
        )


def session_header(context: AgentContext) -> str:
    """Stated once per turn: the task setting, and what the agent may and may not read."""
    reach = context.guard.describe()
    lines = [
        f"Session: {context.session_name} (universe {context.universe}, "
        f"horizon {context.horizon}, IC method {context.ic_method}).",
        f"You may read training data through {reach['readable_through']}. The validation window "
        f"opens {reach['validation_starts']} and the locked holdout opens {reach['test_starts']}; "
        "neither is reachable from any tool you have.",
        f"Budget for this turn: {context.budget.budget.max_tool_calls} tool calls, "
        f"{context.budget.budget.max_evaluations} evaluations.",
    ]
    if context.target_tree is not None:
        lines.append(f"Currently selected factor: {formula_text(context.target_tree)}")
    return "\n".join(lines)


def _langchain_tool_spec(tool: Tool) -> dict[str, Any]:
    """LangChain's provider-neutral tool shape; each adapter reshapes it for its own wire."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.parameters.get("properties", {}),
                "required": tool.parameters.get("required", []),
            },
        },
    }


def _require_langgraph() -> tuple[Any, Any, Any, Any]:
    """Import LangGraph, turning absence into the same clear error the runtime uses.

    Deferred to call time on purpose. LangGraph is a core dependency, but a venv created before
    Phase 10 will not have it, and a module-scope import would make that a startup crash for the
    whole API rather than an unavailable feature.
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import add_messages
    except ImportError as exc:
        from alphalineage.agent.runtime import AgentUnavailable

        raise AgentUnavailable(
            "the agent loop needs langgraph, which is not installed in this environment. "
            'Reinstall the project dependencies (uv pip install -e ".[dev]") and restart. '
            f"({exc})"
        ) from None
    return StateGraph, START, END, add_messages


def _agent_state_type(add_messages: Any) -> Any:
    """The graph's state schema, built once LangGraph's message reducer is in hand.

    The functional ``TypedDict(...)`` form is required rather than a class statement: the reducer
    is only available at call time, and a class body cannot close over it.
    """
    from typing import Annotated, TypedDict

    return TypedDict(  # type: ignore[operator]
        "AgentState",
        {
            "messages": Annotated[list, add_messages],
            "finalizing": bool,
            "stop_reason": str,
            "error": str,
            "final_message": str,
        },
        total=False,
    )


def build_graph(context: AgentContext, *, model: Any) -> tuple[Any, ToolRuntime]:
    """Compile the two-node state machine. Returns the graph and the runtime it will drive."""
    from alphalineage.agent.runtime import invoke_model, usage_from

    state_graph, start_node, end_node, add_messages = _require_langgraph()
    state_type = _agent_state_type(add_messages)
    runtime = ToolRuntime(context)
    bound = model.bind_tools([_langchain_tool_spec(tool) for tool in all_tools()])

    def model_node(state: dict[str, Any]) -> dict[str, Any]:
        try:
            reply = invoke_model(bound, list(state["messages"]))
        except Exception as exc:  # noqa: BLE001 - already redacted upstream
            return {"stop_reason": "failed", "error": str(exc)}
        input_tokens, output_tokens = usage_from(reply)
        context.budget.charge_tokens(input_tokens=input_tokens, output_tokens=output_tokens)
        return {"messages": [reply], "final_message": _text_of(reply)}

    def tools_node(state: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, ToolMessage

        last = list(state["messages"])[-1]
        emitted: list[Any] = []
        stopped = False
        for call in list(getattr(last, "tool_calls", None) or []):
            name = str(call.get("name", ""))
            try:
                payload = runtime.execute(name, call.get("args") or {}).result
            except BudgetExhausted as exc:
                payload, stopped = {"error": str(exc), "budget_exhausted": True}, True
            except Exception as exc:  # noqa: BLE001 - a guard breach fails the turn outright
                emitted.append(
                    ToolMessage(content=str(exc), tool_call_id=str(call.get("id", "")), name=name)
                )
                return {
                    "messages": emitted,
                    "stop_reason": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            emitted.append(
                ToolMessage(content=_json(payload), tool_call_id=str(call.get("id", "")), name=name)
            )
            if stopped:
                break

        reason = context.budget.exhausted_reason()
        if stopped or reason:
            emitted.append(HumanMessage(content=_WRAP_UP.format(reason=reason or "budget spent")))
            return {"messages": emitted, "finalizing": True}
        return {"messages": emitted}

    def route(state: dict[str, Any]) -> str:
        """Model -> tools, or stop. The one place the loop's termination is decided."""
        if state.get("stop_reason") == "failed":
            return end_node
        messages = list(state.get("messages") or [])
        if not messages:
            return end_node
        if not list(getattr(messages[-1], "tool_calls", None) or []):
            return end_node  # answered in prose: done
        if state.get("finalizing"):
            return end_node  # told to wrap up and asked for more work anyway
        return "tools"

    graph = state_graph(state_type)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(start_node, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", end_node: end_node})
    graph.add_edge("tools", "model")
    return graph.compile(), runtime


def run_turn(
    conversation: Conversation,
    context: AgentContext,
    *,
    model: Any,
    message: str,
    max_steps: int = MAX_STEPS,
) -> TurnResult:
    """Run one exchange: the user's message in, one assistant turn out.

    Never raises for model behaviour. A refusal, a malformed tool call, or a provider outage all
    come back as a result with a ``stop_reason``, because a half-finished turn still has a
    transcript worth showing.
    """
    compiled, runtime = build_graph(context, model=model)
    messages, _plan = build_messages(
        conversation,
        system_prompt=SYSTEM_PROMPT,
        header=session_header(context),
        new_message=message,
    )

    result = TurnResult()
    try:
        final_state = compiled.invoke(
            {"messages": messages, "finalizing": False},
            {"recursion_limit": max(4, max_steps * 2)},
        )
    except Exception as exc:  # noqa: BLE001 - includes the graph's own recursion limit
        result.stop_reason = "loop_limit" if "recursion" in str(exc).lower() else "failed"
        result.error = str(exc)
        return _finish(result, runtime, context)

    result.error = str(final_state.get("error") or "")
    result.text = str(final_state.get("final_message") or "")
    if final_state.get("stop_reason") == "failed":
        result.stop_reason = "failed"
    elif context.budget.exhausted_reason():
        result.stop_reason = "budget_exhausted"
    else:
        result.stop_reason = "completed"
    return _finish(result, runtime, context)


def _finish(result: TurnResult, runtime: ToolRuntime, context: AgentContext) -> TurnResult:
    result.calls = [
        ToolCallRecord(
            tool=call.name,
            arguments=call.arguments,
            result=call.result,
            safety=call.safety,
            ok=call.ok,
        )
        for call in runtime.calls
    ]
    result.labels = list(runtime.labels)
    result.proposals = [proposal.to_dict() for proposal in runtime.proposals]
    result.usage = {**context.budget.to_dict(), **context.trials.to_dict()}
    return result


def _text_of(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def _json(payload: dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(payload, default=str)[:20_000]
    except (TypeError, ValueError):
        return str(payload)[:20_000]
