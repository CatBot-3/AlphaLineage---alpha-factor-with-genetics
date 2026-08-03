"""P11-T1 - turning a stored conversation into the messages one model call sees.

Three stages, cheapest first. The ordering is the point: stages 1 and 2 never let a model rewrite
a number, and only stage 3 does. In a thread full of ICs, generalization gaps and lookback counts,
"summary drift" is not a cosmetic risk — a rewritten 0.041 is a wrong fact presented with the same
confidence as a right one.

1. **Recent turns verbatim.** Prose and tool results in full.
2. **Older tool results digested.** Past the recency window each tool result collapses to a
   deterministic one-line digest computed from the result itself. The full text stays on disk and
   stays in the UI; only the model's copy shrinks. Tool results are the bulk of the tokens here,
   so this stage does most of the work at zero risk.
3. **Rolling summary, last resort.** Only if the thread is *still* over budget does one model call
   condense the oldest prose. Most conversations never reach this.

This module is pure: conversation in, message list out. No provider, no I/O, no clock — so the
whole policy is testable without a key.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from alphalineage.agent.conversation import ASSISTANT, USER, Conversation, Turn

#: Roughly four characters per token for English plus JSON. Only used for budgeting.
CHARS_PER_TOKEN = 4

#: How many of the most recent turns keep their tool results in full.
DEFAULT_VERBATIM_TURNS = 3
#: Character budget for the replayed history, excluding the system prompt and the new message.
DEFAULT_HISTORY_BUDGET_CHARS = 24_000
#: A single tool result larger than this is clipped even inside the verbatim window; one runaway
#: payload should not evict the rest of the conversation.
MAX_VERBATIM_RESULT_CHARS = 6_000


@dataclass
class ReplayPlan:
    """What compaction decided, surfaced so the UI and tests can see it rather than infer it."""

    verbatim_turns: int = 0
    digested_turns: int = 0
    summarised_turns: int = 0
    characters: int = 0
    summary_used: bool = False

    @property
    def estimated_tokens(self) -> int:
        return self.characters // CHARS_PER_TOKEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "verbatim_turns": self.verbatim_turns,
            "digested_turns": self.digested_turns,
            "summarised_turns": self.summarised_turns,
            "characters": self.characters,
            "estimated_tokens": self.estimated_tokens,
            "summary_used": self.summary_used,
        }


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters; full result kept on disk]"


def render_turn(turn: Turn, *, verbatim: bool) -> str:
    """One turn as plain text for replay.

    ``verbatim`` keeps tool results; otherwise each collapses to its deterministic digest.
    """
    import json

    if turn.role == USER:
        return f"User: {turn.text}"

    parts: list[str] = []
    if turn.text:
        parts.append(f"Assistant: {turn.text}")
    for call in turn.calls:
        if verbatim:
            try:
                payload = json.dumps(call.result, default=str)
            except (TypeError, ValueError):
                payload = str(call.result)
            parts.append(
                f"[tool {call.tool}] args={json.dumps(call.arguments, default=str)[:600]} "
                f"result={_clip(payload, MAX_VERBATIM_RESULT_CHARS)}"
            )
        else:
            parts.append(f"[tool] {call.digest()}")
    if turn.labels:
        names = ", ".join(str(item.get("name", "")) for item in turn.labels)
        parts.append(f"[labels attached] {names}")
    return "\n".join(parts) if parts else "Assistant: (no reply)"


def plan_replay(
    conversation: Conversation,
    *,
    verbatim_turns: int = DEFAULT_VERBATIM_TURNS,
    budget_chars: int = DEFAULT_HISTORY_BUDGET_CHARS,
) -> tuple[list[str], ReplayPlan]:
    """Render the history under budget, returning the blocks and what was done to get there.

    Walks backwards from the most recent turn so the newest material is always kept, which is
    what an agent actually needs to continue: the last thing it tried and what came back.
    """
    plan = ReplayPlan()
    blocks: list[str] = []
    used = 0
    turns = conversation.turns[conversation.summarised_through :]

    for offset, turn in enumerate(reversed(turns)):
        verbatim = offset < verbatim_turns
        text = render_turn(turn, verbatim=verbatim)
        if used + len(text) > budget_chars and blocks:
            if verbatim:  # try again digested before giving up on the turn entirely
                text = render_turn(turn, verbatim=False)
                verbatim = False
            if used + len(text) > budget_chars:
                plan.summarised_turns = len(turns) - offset
                break
        blocks.append(text)
        used += len(text)
        if verbatim:
            plan.verbatim_turns += 1
        else:
            plan.digested_turns += 1

    blocks.reverse()
    if conversation.summary:
        header = f"Earlier in this conversation (summarised):\n{conversation.summary}"
        blocks.insert(0, header)
        used += len(header)
        plan.summary_used = True
        plan.summarised_turns += conversation.summarised_through
    plan.characters = used
    return blocks, plan


def build_messages(
    conversation: Conversation,
    *,
    system_prompt: str,
    header: str,
    new_message: str,
    verbatim_turns: int = DEFAULT_VERBATIM_TURNS,
    budget_chars: int = DEFAULT_HISTORY_BUDGET_CHARS,
    message_factory: Callable[[str, str], Any] | None = None,
) -> tuple[list[Any], ReplayPlan]:
    """Assemble the message list for one model call.

    ``message_factory(role, text)`` builds a provider message; the default constructs LangChain
    messages lazily so this module stays importable without LangChain (see docs/AGENT.md §7).
    """
    factory = message_factory or _default_message_factory()
    blocks, plan = plan_replay(
        conversation, verbatim_turns=verbatim_turns, budget_chars=budget_chars
    )

    messages: list[Any] = [factory("system", system_prompt)]
    opening = header
    if blocks:
        opening = f"{header}\n\n--- conversation so far ---\n" + "\n\n".join(blocks)
    messages.append(factory("user", opening))
    if new_message.strip():
        messages.append(factory("user", new_message.strip()))
    return messages, plan


def _default_message_factory() -> Callable[[str, str], Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    def build(role: str, text: str) -> Any:
        return SystemMessage(content=text) if role == "system" else HumanMessage(content=text)

    return build


# --- stage 3 -----------------------------------------------------------------------
SUMMARY_PROMPT = """\
Condense the following research conversation into at most 200 words. Preserve, exactly as
written: every expression tried, every numeric result, and every conclusion reached. Drop
pleasantries and restatements. Do not round, re-derive, or reinterpret any number — copy it. Do
not add analysis of your own."""


def needs_summary(
    conversation: Conversation,
    *,
    budget_chars: int = DEFAULT_HISTORY_BUDGET_CHARS,
    verbatim_turns: int = DEFAULT_VERBATIM_TURNS,
) -> bool:
    """True when digesting was not enough and prose has started being dropped."""
    _, plan = plan_replay(conversation, verbatim_turns=verbatim_turns, budget_chars=budget_chars)
    return plan.summarised_turns > conversation.summarised_through


def summarise(
    conversation: Conversation,
    *,
    invoke: Callable[[str], str],
    keep_recent: int = DEFAULT_VERBATIM_TURNS * 2,
) -> Conversation:
    """Fold the oldest turns into ``summary`` using one model call.

    ``invoke(prompt) -> text`` is injected so this is testable without a provider. The
    conversation is mutated and returned; the caller persists it.
    """
    cutoff = max(0, len(conversation.turns) - keep_recent)
    if cutoff <= conversation.summarised_through:
        return conversation
    pending = conversation.turns[conversation.summarised_through : cutoff]
    if not pending:
        return conversation

    body = "\n\n".join(render_turn(turn, verbatim=False) for turn in pending)
    previous = f"Existing summary:\n{conversation.summary}\n\n" if conversation.summary else ""
    text = invoke(f"{SUMMARY_PROMPT}\n\n{previous}Conversation:\n{body}")
    conversation.summary = text.strip() or conversation.summary
    conversation.summarised_through = cutoff
    return conversation


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_HISTORY_BUDGET_CHARS",
    "DEFAULT_VERBATIM_TURNS",
    "SUMMARY_PROMPT",
    "ASSISTANT",
    "USER",
    "ReplayPlan",
    "build_messages",
    "needs_summary",
    "plan_replay",
    "render_turn",
    "summarise",
]
