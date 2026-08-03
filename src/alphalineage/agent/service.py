"""P11 - orchestration: one message in, one assistant turn out, and the approval gates.

Three entry points.

:func:`build_context` assembles the world a turn may see from a stored session.

:func:`send_message` runs one exchange and appends it to the persisted conversation. This is the
only place a turn is created.

:func:`promote` and :func:`apply_config` are the gates. Nothing the agent stages takes effect
until one of them is called by a human action. Promotion is the heavier of the two: it validates
the factor properly and folds the conversation's trial spend into the session before the round
exists, because a round is finalizable the moment it does.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from alphalineage.agent.context import AgentContext, context_from_session, target_from_round
from alphalineage.agent.conversation import USER, Conversation, ConversationStore, Turn
from alphalineage.agent.guards import (
    Budget,
    BudgetLedger,
    GuardViolation,
    validate_config_patch,
)
from alphalineage.agent.runtime import AgentUnavailable, build_chat_model, supports_tool_calling
from alphalineage.core.panel import Panel
from alphalineage.explain.credentials import load_config
from alphalineage.explain.providers import LLMError


class AgentError(ValueError):
    """A turn could not be assembled, or a proposal could not be applied."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_context(
    session_id: str,
    *,
    panel: Panel,
    round_index: int | None = None,
    budget: Budget | None = None,
) -> AgentContext:
    """Assemble the agent's world from a stored session and its latest completed segment."""
    from alphalineage.api import sessions as sessions_module

    if not sessions_module.exists(session_id):
        raise AgentError(f"session {session_id!r} not found")
    session = sessions_module.load_session(session_id)

    rounds = sessions_module.list_rounds(session_id)
    if round_index is None:
        completed = [item for item in rounds if item.get("status") == "done"]
        round_index = int((completed or rounds or [{"index": 0}])[-1].get("index", 0))
    result = sessions_module.load_round(session_id, int(round_index))
    tree, metrics = target_from_round(result or {})

    expanded = tree
    if tree is not None:
        from alphalineage.agent.vocabulary import expand_tree

        expanded = expand_tree(
            tree, None, pinned_specs=session.get("formula_revisions") or session.get("operators")
        )
    try:
        return context_from_session(
            session,
            panel=panel,
            target_tree=tree,
            target_expanded=expanded,
            target_label=f"{session.get('name', session_id)} — round {round_index}",
            target_metrics=metrics,
            budget=BudgetLedger(budget=budget or Budget()),
        )
    except GuardViolation as exc:
        raise AgentError(str(exc)) from None


def send_message(
    session_id: str,
    context: AgentContext,
    *,
    message: str,
    provider: str = "",
    model: str = "",
    base_url: str = "",
    store: ConversationStore | None = None,
    chat_model: Any = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Run one exchange and append both turns to the conversation.

    ``chat_model`` is injectable so the whole pipeline can be exercised without a provider.
    """
    from alphalineage.agent.graph import run_turn

    conversations = store or ConversationStore()
    conversation = conversations.load_or_create(session_id, context.session_name)

    llm = load_config(provider=provider, model=model, base_url=base_url)
    if chat_model is None:
        if not llm.key.is_set:
            raise LLMError(
                f"no API key configured for provider '{llm.provider}'. Add one in Settings "
                "or set the provider's environment variable."
            )
        supported, reason = supports_tool_calling(llm)
        if not supported:
            raise LLMError(reason)
        chat_model = build_chat_model(llm)

    conversation.add(Turn(role=USER, text=message))
    result = run_turn(conversation, context, model=chat_model, message=message)
    turn = result.to_turn()
    turn.usage = {**turn.usage, "provider": llm.provider, "model": llm.model}
    conversation.add(turn)

    if persist:
        conversations.save(conversation)
    return {
        "session_id": session_id,
        "turn": turn.to_dict(),
        "turn_index": len(conversation.turns) - 1,
        "conversation": conversation.summary_view(),
    }


# --- the gates -------------------------------------------------------------------
def _find_proposal(
    conversation: Conversation, proposal_id: str, kind: str
) -> tuple[int, dict[str, Any]]:
    found = conversation.proposal(proposal_id)
    if found is None:
        raise AgentError(f"proposal {proposal_id!r} not found in this conversation")
    turn_index, proposal = found
    if proposal.get("applied"):
        raise AgentError("this proposal has already been applied")
    if proposal.get("kind") != kind:
        raise AgentError(
            f"proposal {proposal_id!r} is a {proposal.get('kind')!r} proposal, not {kind!r}"
        )
    return turn_index, proposal


def plan_promotion(
    session_id: str, proposal_id: str, *, store: ConversationStore | None = None
) -> dict[str, Any]:
    """Validate that a promotion can proceed, so a bad id is a 400 rather than a failed job."""
    conversations = store or ConversationStore()
    conversation = conversations.get(session_id)
    if conversation is None:
        raise AgentError("this session has no conversation yet")
    turn_index, proposal = _find_proposal(conversation, proposal_id, "factor")
    return {"turn_index": turn_index, "name": (proposal.get("payload") or {}).get("name")}


def promote(
    session_id: str,
    proposal_id: str,
    *,
    panel: Panel,
    store: ConversationStore | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Validate a staged factor and write it as a new round."""
    from alphalineage.agent.promote import PromotionError, promote_factor

    conversations = store or ConversationStore()
    conversation = conversations.get(session_id)
    if conversation is None:
        raise AgentError("this session has no conversation yet")
    turn_index, proposal = _find_proposal(conversation, proposal_id, "factor")
    payload = proposal.get("payload") or {}

    parent = None
    from alphalineage.api import sessions as sessions_module

    for summary in reversed(sessions_module.list_rounds(session_id)):
        if summary.get("origin") != "agent":
            parent = int(summary.get("index", 0))
            break

    try:
        summary = promote_factor(
            session_id,
            tree_payload=payload.get("tree") or {},
            name=str(payload.get("name") or "Agent candidate"),
            rationale=str(proposal.get("rationale") or ""),
            panel=panel,
            conversation_id=session_id,
            turn_index=turn_index,
            agent_trials=conversation.total_trials,
            parent_round_index=parent,
            progress=progress,
        )
    except PromotionError as exc:
        raise AgentError(str(exc)) from None

    proposal["applied"] = True
    proposal["round_index"] = summary.get("index")
    conversations.save(conversation)
    return {
        "promoted": proposal_id,
        "round": summary,
        "trials_folded_into_session": conversation.total_trials,
        "note": (
            "Validated on the session's validation window and written as a new round. Its "
            "search cost was added to the session's cumulative trial count, so later deflated "
            "statistics account for it."
        ),
    }


def apply_config(
    session_id: str, proposal_id: str, *, store: ConversationStore | None = None
) -> dict[str, Any]:
    """Merge a staged patch into the session's stored config for the next segment.

    Re-validated here rather than trusted from the transcript: the session may have moved on
    since the turn, and a patch that was legal then may not be legal now.
    """
    from alphalineage.api import sessions as sessions_module

    conversations = store or ConversationStore()
    conversation = conversations.get(session_id)
    if conversation is None:
        raise AgentError("this session has no conversation yet")
    _, proposal = _find_proposal(conversation, proposal_id, "config")
    patch = (proposal.get("payload") or {}).get("patch")
    if not isinstance(patch, dict) or not patch:
        raise AgentError("this proposal carries no configuration patch")
    if not sessions_module.exists(session_id):
        raise AgentError("the session this conversation belongs to no longer exists")

    session = sessions_module.load_session(session_id)
    result = validate_config_patch(patch, dict(session.get("config") or {}))
    if not result.ok:
        raise AgentError("; ".join(result.errors))

    session["config"] = result.merged
    session["updated_at"] = _now()
    sessions_module.save_session(session)

    proposal["applied"] = True
    conversations.save(conversation)
    return {
        "applied": "config",
        "session_id": session_id,
        "diff": result.diff,
        "next_step": "Continue the session to run a segment with the new configuration.",
    }


def unavailable_reason() -> str:
    """Why the agent cannot run, or an empty string when it can."""
    try:
        from alphalineage.agent.runtime import _require_langchain

        _require_langchain()
    except AgentUnavailable as exc:
        return str(exc)
    try:
        import langgraph.graph  # noqa: F401
    except ImportError as exc:
        return (
            "the agent needs langgraph, which is not installed in this environment. "
            'Reinstall the project dependencies (uv pip install -e ".[dev]") and restart. '
            f"({exc})"
        )
    return ""
