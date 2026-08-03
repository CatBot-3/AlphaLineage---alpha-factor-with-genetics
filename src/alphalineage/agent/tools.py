"""P11-T3 - the tool registry. This list is the agent's entire capability.

Anything not here is unreachable: there is no fallback, no passthrough, and no generic "run this"
tool. ``GET /agent/tools`` returns the registry verbatim so a user can read exactly what they are
authorizing before a key is spent.

One registry, not two. Phase 10 split the tools into an "explainer" set and an "improver" set,
which forced the user to classify their own question before asking it. The model now gets
everything and decides: "what does this measure" reaches for the read tools, "make it better"
reaches for evaluate and propose. Same blast radius, one door.

Four safety classes:

* **read** — touches no market data, so it is free and unbudgeted. ``validate_expression`` is the
  important one: it is the model's *compiler*, letting it iterate to a well-typed expression
  without spending a single trial. Making that free is what keeps the evaluation budget spent on
  real hypotheses instead of syntax errors.
* **evaluate** — scores a candidate on the inner split inside the training window. Budgeted and
  trial-counted. It takes no date, universe, or split argument; there is no parameter through
  which the validation window or the locked holdout can be addressed.
* **propose** — stages a change for a human to approve. Mutates nothing.
* **annotate** — attaches structured labels to the current turn. Writes only to the conversation,
  so it needs no approval. This is what replaced the hand-written aspect dictionary.

Handlers return JSON-serializable dicts. Errors are *returned*, not raised, so the model can read
what it did wrong and correct it — a rejected patch or a mistyped expression should cost one tool
call, not the whole turn.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from alphalineage.agent.context import AgentContext
from alphalineage.agent.evaluate import correlation_with_target, score_candidate
from alphalineage.agent.expressions import ExpressionError, parse_expression
from alphalineage.agent.guards import BudgetExhausted, validate_config_patch
from alphalineage.agent.vocabulary import operator_listing
from alphalineage.core.tree import Node
from alphalineage.core.tree import to_dict as tree_to_dict
from alphalineage.explain.anatomy import formula_text, measure, render_markdown

READ = "read"
EVALUATE = "evaluate"
PROPOSE = "propose"
ANNOTATE = "annotate"


@dataclass(frozen=True)
class Tool:
    """One capability, with the JSON schema the model is shown."""

    name: str
    description: str
    parameters: dict[str, Any]
    safety: str
    handler: Callable[[ToolRuntime, dict[str, Any]], dict[str, Any]]

    def spec(self) -> dict[str, Any]:
        """Provider-neutral description; adapters reshape this per wire format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "safety": self.safety,
        }


@dataclass
class Proposal:
    """A staged change. Inert until a human approves it."""

    id: str
    kind: str  # "factor" | "config"
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "rationale": self.rationale,
            "payload": self.payload,
            "applied": self.applied,
        }


@dataclass
class ToolCall:
    """One executed call, recorded verbatim for the transcript."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    safety: str
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "safety": self.safety,
            "ok": self.ok,
        }


class ToolRuntime:
    """Executes tools against one :class:`AgentContext`, charging budgets as it goes."""

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self.calls: list[ToolCall] = []
        self.proposals: list[Proposal] = []
        self.labels: list[dict[str, Any]] = []
        self._allowed_operators: set[str] | None = None

    @property
    def allowed_operators(self) -> set[str]:
        """Names the agent may use, restricted to the session's enabled search space."""
        if self._allowed_operators is None:
            self._allowed_operators = {
                str(item["name"]) for item in operator_listing(self.context.config)
            }
        return self._allowed_operators

    def parse(self, text: str) -> Node:
        """Parse the DSL text form, restricted to this session's vocabulary."""
        return parse_expression(text, allowed=self.allowed_operators)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        """Run one tool. Never raises for model-caused errors; raises only on budget exhaustion."""
        tool = TOOLS.get(name)
        if tool is None:
            call = ToolCall(
                name=name,
                arguments=dict(arguments or {}),
                result={"error": f"unknown tool {name!r}", "available": sorted(TOOLS)},
                safety="unknown",
                ok=False,
            )
            self.calls.append(call)
            return call

        self.context.budget.charge_tool_call()  # BudgetExhausted propagates: the turn stops
        try:
            result = tool.handler(self, dict(arguments or {}))
            ok = "error" not in result
        except BudgetExhausted:
            raise
        except ExpressionError as exc:
            result, ok = exc.to_dict(), False
        except (ValueError, KeyError, TypeError) as exc:
            result, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
        result = {**result, "budget_remaining": self.context.budget.remaining()}
        call = ToolCall(
            name=name,
            arguments=dict(arguments or {}),
            result=result,
            safety=tool.safety,
            ok=ok,
        )
        self.calls.append(call)
        return call

    def stage(self, kind: str, rationale: str, payload: dict[str, Any]) -> Proposal:
        proposal = Proposal(
            id=uuid.uuid4().hex[:12], kind=kind, rationale=rationale, payload=payload
        )
        self.proposals.append(proposal)
        return proposal


# --- class R: read-only, free ----------------------------------------------------
def _list_operators(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    category = str(args.get("category") or "").strip()
    items = operator_listing(runtime.context.config)
    if category:
        items = [item for item in items if item.get("category") == category]
    return {
        "operators": [
            {
                "name": item["name"],
                "signature": (
                    f"{item['name']}({', '.join(item['arg_types'])}) -> {item['out_type']}"
                ),
                "category": item["category"],
                "description": item["description"],
            }
            for item in items
        ],
        "count": len(items),
        "note": "These are the only names that may appear in an expression.",
    }


def _validate_expression(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    tree = runtime.parse(str(args.get("expression", "")))
    measurement = measure(tree, horizon=runtime.context.horizon)
    return {
        "valid": True,
        "canonical": formula_text(tree),
        "out_type": tree.out_type.value,
        "nodes": tree.size(),
        "depth": tree.depth(),
        "effective_lookback_bars": measurement.windows.effective_lookback_bars,
    }


def _measure_expression(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    tree = runtime.parse(str(args.get("expression", "")))
    measurement = measure(tree, horizon=runtime.context.horizon)
    return {
        "expression": formula_text(tree),
        "measurement": render_markdown(measurement),
        "note": (
            "These numbers are computed from the tree's shape, not estimated. The effective "
            "lookback in particular is the compounded total across nested windows, which is "
            "usually much larger than the biggest window in the expression."
        ),
    }


def _get_current_factor(runtime: ToolRuntime, _: dict[str, Any]) -> dict[str, Any]:
    context = runtime.context
    if context.target_tree is None:
        return {"error": "this session has no selected factor yet"}
    measurement = measure(
        context.target_tree, expanded=context.target_expanded, horizon=context.horizon
    )
    return {
        "label": context.target_label,
        "expression": formula_text(context.target_tree),
        "expanded_expression": formula_text(context.target_expanded or context.target_tree),
        "measurement": render_markdown(measurement),
        "recorded_metrics": context.target_metrics,
        "metrics_note": (
            "Training metrics are in-sample. Validation metrics come from the session's "
            "validation window and are NOT a holdout result. Do not treat either as evidence "
            "of live performance."
        ),
    }


def _get_search_trajectory(runtime: ToolRuntime, _: dict[str, Any]) -> dict[str, Any]:
    from alphalineage.explain.corpus import build_corpus

    session_id = runtime.context.session_id
    documents = [
        doc
        for doc in build_corpus(include=["genetics", "round", "session"])
        if session_id and session_id in doc.id
    ]
    if not documents:
        return {"trajectory": "", "note": "no stored segments for this session yet"}
    return {
        "documents": [{"id": doc.id, "text": doc.text} for doc in documents],
        "note": (
            "Fitness values here are training-window objectives from the search. They are not "
            "out-of-sample evidence."
        ),
    }


def _get_session_config(runtime: ToolRuntime, _: dict[str, Any]) -> dict[str, Any]:
    from alphalineage.agent.guards import PROTECTED_KEYS, TUNABLE_KEYS

    config = runtime.context.config
    return {
        "config": {key: config.get(key) for key in sorted(config)},
        "tunable_keys": sorted(TUNABLE_KEYS),
        "protected_keys": {key: reason for key, reason in sorted(PROTECTED_KEYS.items())},
        "context": runtime.context.describe(),
    }


def _search_workspace(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Retrieval on demand, replacing Phase 9's automatic prompt stuffing.

    The model asks when it decides it needs history, which means a question that does not need
    the archive does not pay for it.
    """
    from alphalineage.explain.corpus import build_corpus
    from alphalineage.explain.retrieval import Query, retrieve

    query = str(args.get("query") or "").strip()
    if not query:
        return {"error": "a query is required"}
    kinds = args.get("kinds")
    include = [str(k) for k in kinds] if isinstance(kinds, list) and kinds else None
    documents = build_corpus(include=include)
    hits = retrieve(
        documents,
        Query(text=query, universe=runtime.context.universe),
        budget_chars=int(args.get("budget_chars") or 8000),
        max_documents=int(args.get("limit") or 8),
    )
    return {
        "results": [
            {"id": hit.document.id, "kind": hit.document.kind, "text": hit.document.text}
            for hit in hits
        ],
        "count": len(hits),
    }


def _list_agent_rounds(runtime: ToolRuntime, _: dict[str, Any]) -> dict[str, Any]:
    """The model's own prior rounds, so it can build on them instead of restarting each time."""
    from alphalineage.api import sessions as sessions_module

    session_id = runtime.context.session_id
    if not session_id or not sessions_module.exists(session_id):
        return {"rounds": [], "count": 0}
    rows = []
    for summary in sessions_module.list_rounds(session_id):
        if summary.get("origin") != "agent":
            continue
        rows.append(
            {
                "index": summary.get("index"),
                "expression": summary.get("agent_expression"),
                "parent_round_index": summary.get("parent_round_index"),
                "validation_oriented_ic": (summary.get("selection") or {})
                .get("validation_metrics", {})
                .get("oriented_ic"),
                "evidence_status": summary.get("evidence_status"),
            }
        )
    return {
        "rounds": rows,
        "count": len(rows),
        "note": (
            "Rounds you authored earlier in this session. Build on them rather than repeating them."
        ),
    }


# --- class E: evaluation, budgeted -----------------------------------------------
def _evaluate_expression(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    context = runtime.context
    context.budget.charge_evaluation()
    tree = runtime.parse(str(args.get("expression", "")))
    expression = formula_text(tree)
    context.trials.record(expression)
    score = score_candidate(tree, context, note=str(args.get("note") or ""), expression=expression)
    payload = score.to_dict()
    payload["generalization_gap"] = score.generalization_gap
    payload["correlation_with_current_factor"] = correlation_with_target(tree, context)
    payload["trials_spent_so_far"] = context.trials.count
    return payload


# --- class P: proposals, staged --------------------------------------------------
def _propose_factor(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    tree = runtime.parse(str(args.get("expression", "")))
    name = str(args.get("name") or "").strip() or "Agent candidate"
    rationale = str(args.get("rationale") or "").strip()
    if not rationale:
        return {"error": "a rationale is required: say why this candidate is worth keeping"}
    proposal = runtime.stage(
        "factor",
        rationale,
        {
            "name": name[:120],
            "expression": formula_text(tree),
            "tree": tree_to_dict(tree),
        },
    )
    return {
        "staged": True,
        "proposal_id": proposal.id,
        "note": (
            "Staged for human approval. Promoting it runs the session's real validation pass "
            "and creates a new round; nothing has happened yet."
        ),
    }


def _propose_config_patch(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    patch = args.get("patch")
    rationale = str(args.get("rationale") or "").strip()
    if not rationale:
        return {"error": "a rationale is required: ground the change in the search trajectory"}
    result = validate_config_patch(patch if isinstance(patch, dict) else {}, runtime.context.config)
    if not result.ok:
        return {"error": "; ".join(result.errors), "rejected_patch": result.patch}
    proposal = runtime.stage("config", rationale, {"patch": result.patch, "diff": result.diff})
    return {
        "staged": True,
        "proposal_id": proposal.id,
        "diff": result.diff,
        "note": "Staged for human approval. The session configuration is unchanged.",
    }


# --- class A: annotations --------------------------------------------------------
def _annotate_factor(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Attach structured labels to this turn.

    This replaced a hand-written dictionary of fixed aspect names. The vocabulary is now open:
    if a factor reads something the old list never had a word for, the model can just say so.
    """
    raw = args.get("labels")
    if not isinstance(raw, list) or not raw:
        return {"error": "labels must be a non-empty list"}
    labels: list[dict[str, Any]] = []
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        labels.append(
            {
                "name": name[:60],
                "evidence": str(item.get("evidence") or "")[:400],
                "reading": str(item.get("reading") or "")[:600],
                "risk": str(item.get("risk") or "")[:600],
            }
        )
    if not labels:
        return {"error": "no usable labels: each needs at least a name"}
    runtime.labels.extend(labels)
    return {
        "attached": len(labels),
        "names": [item["name"] for item in labels],
        "note": "Shown alongside your answer. These are your labels, not computed facts.",
    }


# --- the registry ----------------------------------------------------------------
_EXPRESSION_PARAM = {
    "type": "string",
    "description": "A DSL expression in function-call form, e.g. rank(ts_std(returns, 20)).",
}

TOOL_LIST: tuple[Tool, ...] = (
    Tool(
        name="get_current_factor",
        description=(
            "The factor currently selected in this session: its expression, its computed shape, "
            "and the metrics recorded for it, each labelled with the split it came from. Start "
            "here when asked about 'this factor'."
        ),
        parameters={"type": "object", "properties": {}},
        safety=READ,
        handler=_get_current_factor,
    ),
    Tool(
        name="measure_expression",
        description=(
            "Compute an expression's shape: compounded effective lookback, window profile, "
            "depth and size, which data fields it reads, unit consistency, and structural "
            "diagnostics. Reads no market data. Use it rather than working the lookback out by "
            "eye — nested windows compound and the total is easy to underestimate."
        ),
        parameters={
            "type": "object",
            "properties": {"expression": _EXPRESSION_PARAM},
            "required": ["expression"],
        },
        safety=READ,
        handler=_measure_expression,
    ),
    Tool(
        name="list_operators",
        description=(
            "List every operator and data field this session's search space allows, with its "
            "signature. These are the only names an expression may use."
        ),
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional filter, e.g. time_series, cross_sectional, data.",
                }
            },
        },
        safety=READ,
        handler=_list_operators,
    ),
    Tool(
        name="validate_expression",
        description=(
            "Parse and type-check an expression without evaluating it. Free and unbudgeted: use "
            "it as a compiler to fix syntax and typing before spending an evaluation."
        ),
        parameters={
            "type": "object",
            "properties": {"expression": _EXPRESSION_PARAM},
            "required": ["expression"],
        },
        safety=READ,
        handler=_validate_expression,
    ),
    Tool(
        name="get_search_trajectory",
        description=(
            "How the genetic search behaved: the fitness curve, when improvement stalled, "
            "population diversity, the genetic-operator mix, and per-formula parameter search."
        ),
        parameters={"type": "object", "properties": {}},
        safety=READ,
        handler=_get_search_trajectory,
    ),
    Tool(
        name="get_session_config",
        description=(
            "The session's current search configuration, which keys may be tuned, and which are "
            "protected along with the reason."
        ),
        parameters={"type": "object", "properties": {}},
        safety=READ,
        handler=_get_session_config,
    ),
    Tool(
        name="search_workspace",
        description=(
            "Search this workspace's own history: past sessions, completed segments, saved "
            "factors, operators, and the indicator catalog. Use it when you need background you "
            "were not already given."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you are looking for."},
                "kinds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional filter: session, round, genetics, factor, "
                    "indicator, primitive.",
                },
                "limit": {"type": "integer", "description": "Maximum documents (default 8)."},
            },
            "required": ["query"],
        },
        safety=READ,
        handler=_search_workspace,
    ),
    Tool(
        name="list_agent_rounds",
        description=(
            "The rounds you created earlier in this session, with their validation results. "
            "Check this before proposing, so you build on your own work instead of repeating it."
        ),
        parameters={"type": "object", "properties": {}},
        safety=READ,
        handler=_list_agent_rounds,
    ),
    Tool(
        name="evaluate_expression",
        description=(
            "Score a candidate on an inner holdout carved out of the TRAINING window, with "
            "costs. Returns inner-train and inner-holdout IC, the generalization gap, and the "
            "rank correlation with the current factor. This is NOT validation evidence and it "
            "cannot read the validation or holdout splits. Each call costs one trial from a "
            "limited budget, so validate first and evaluate only real hypotheses."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": _EXPRESSION_PARAM,
                "note": {
                    "type": "string",
                    "description": "What you are testing with this candidate, for the transcript.",
                },
            },
            "required": ["expression"],
        },
        safety=EVALUATE,
        handler=_evaluate_expression,
    ),
    Tool(
        name="propose_factor",
        description=(
            "Stage a candidate factor for the human to review. If they promote it, the session's "
            "real validation pass runs and it becomes a new round you can build on later."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": _EXPRESSION_PARAM,
                "name": {"type": "string", "description": "A short descriptive name."},
                "rationale": {
                    "type": "string",
                    "description": "Why this is worth keeping, referencing the numbers you saw.",
                },
            },
            "required": ["expression", "name", "rationale"],
        },
        safety=PROPOSE,
        handler=_propose_factor,
    ),
    Tool(
        name="propose_config_patch",
        description=(
            "Stage a change to the next segment's search configuration, as a partial object of "
            "config keys. Only tunable keys are accepted; protected keys are refused with the "
            "reason. Staging changes nothing until the human approves."
        ),
        parameters={
            "type": "object",
            "properties": {
                "patch": {
                    "type": "object",
                    "description": 'Partial config, e.g. {"generations": 40}.',
                },
                "rationale": {
                    "type": "string",
                    "description": "Ground the change in the observed search trajectory.",
                },
            },
            "required": ["patch", "rationale"],
        },
        safety=PROPOSE,
        handler=_propose_config_patch,
    ),
    Tool(
        name="annotate_factor",
        description=(
            "Attach structured labels describing what a factor measures — one entry per property "
            "you identify, with the evidence for it and how it could fail. The vocabulary is "
            "yours to choose; there is no fixed list. Use this when asked what a factor does, so "
            "the labels render alongside your answer."
        ),
        parameters={
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "description": "One entry per property you identify.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Short label."},
                            "evidence": {
                                "type": "string",
                                "description": "The part of the expression that provides it.",
                            },
                            "reading": {
                                "type": "string",
                                "description": "What it contributes to the signal's meaning.",
                            },
                            "risk": {
                                "type": "string",
                                "description": "How it could fail or become crowded.",
                            },
                        },
                        "required": ["name"],
                    },
                }
            },
            "required": ["labels"],
        },
        safety=ANNOTATE,
        handler=_annotate_factor,
    ),
)

TOOLS: dict[str, Tool] = {tool.name: tool for tool in TOOL_LIST}


def catalog(kinds: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """The public tool listing. ``kinds`` optionally filters by safety class."""
    return [tool.spec() for tool in TOOL_LIST if not kinds or tool.safety in kinds]


def all_tools() -> list[Tool]:
    """Every tool. There is one toolset now — the model decides what a question needs."""
    return list(TOOL_LIST)
