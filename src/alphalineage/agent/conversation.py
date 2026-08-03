"""P11-T1 - the conversation: one thread, bound to one training session.

A conversation belongs to exactly one training session, takes that session's name, and lives at
``data_cache/sessions/{session_id}/conversation.json``. Storing it *inside* the session directory
is the whole trick behind "deleting the training session deletes the LLM session": there is no
separate cleanup path that can be forgotten, because removing the directory removes the thread,
its memory, and its tool-call history together.

A turn is the unit the UI renders and the unit memory compaction operates on. Assistant turns
carry their tool calls verbatim — that record is what lets a user trace a suggestion back to the
number that justified it, so it is never trimmed on disk, only on replay to the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alphalineage.data import paths
from alphalineage.data.identifiers import atomic_write_text

DISCLAIMER = "Not investment advice. Research output only."
SCHEMA_VERSION = 1

USER = "user"
ASSISTANT = "assistant"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def conversation_path(session_id: str) -> Path:
    """Inside the session directory, so session deletion takes the conversation with it."""
    from alphalineage.api import sessions as sessions_module

    return sessions_module.session_dir(session_id) / "conversation.json"


@dataclass
class ToolCallRecord:
    """One executed tool call, kept in full for the transcript."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    safety: str = "read"
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallRecord:
        return cls(
            tool=str(data.get("tool", "")),
            arguments=dict(data.get("arguments") or {}),
            result=dict(data.get("result") or {}),
            safety=str(data.get("safety", "read")),
            ok=bool(data.get("ok", True)),
        )

    def digest(self) -> str:
        """A one-line stand-in used when this call is too old to replay in full.

        Deterministic — it reads numbers that are already in the result rather than asking a
        model to summarize them, so a value can never drift between what the UI shows and what
        the model is later told.
        """
        if not self.ok:
            return f"{self.tool}: error — {str(self.result.get('error', 'failed'))[:120]}"
        if self.tool == "evaluate_expression":
            holdout = self.result.get("inner_holdout") or {}
            parts = [f"inner-holdout IC {_num(holdout.get('oriented_ic'))}"]
            if self.result.get("generalization_gap") is not None:
                parts.append(f"gap {_num(self.result['generalization_gap'])}")
            if self.result.get("correlation_with_current_factor") is not None:
                parts.append(f"corr {_num(self.result['correlation_with_current_factor'])}")
            expression = str(self.arguments.get("expression", ""))[:120]
            return f"{self.tool}({expression}): " + ", ".join(parts)
        if self.tool == "validate_expression":
            return f"{self.tool}: valid, {self.result.get('nodes')} nodes"
        if self.tool == "list_operators":
            return f"{self.tool}: {self.result.get('count')} operators listed"
        if self.result.get("staged"):
            return f"{self.tool}: staged proposal {self.result.get('proposal_id')}"
        keys = ", ".join(sorted(k for k in self.result if k != "budget_remaining")[:6])
        return f"{self.tool}: returned {keys or 'ok'}"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


@dataclass
class Turn:
    """One message. Assistant turns carry the work that produced them."""

    role: str
    text: str = ""
    at: str = field(default_factory=_now)
    calls: list[ToolCallRecord] = field(default_factory=list)
    #: Structured labels the model attached via ``annotate_factor`` — this is what replaced the
    #: hand-written aspect chips.
    labels: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "at": self.at,
            "calls": [call.to_dict() for call in self.calls],
            "labels": list(self.labels),
            "proposals": list(self.proposals),
            "usage": self.usage,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        return cls(
            role=str(data.get("role", ASSISTANT)),
            text=str(data.get("text", "")),
            at=str(data.get("at", "")),
            calls=[ToolCallRecord.from_dict(c) for c in data.get("calls") or []],
            labels=list(data.get("labels") or []),
            proposals=list(data.get("proposals") or []),
            usage=dict(data.get("usage") or {}),
            stop_reason=str(data.get("stop_reason", "")),
            error=str(data.get("error", "")),
        )


@dataclass
class Conversation:
    """The whole thread for one training session."""

    session_id: str
    session_name: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    turns: list[Turn] = field(default_factory=list)
    #: Written only when stage-3 compaction fires; empty in the common case.
    summary: str = ""
    #: Index of the last turn already folded into ``summary``.
    summarised_through: int = 0
    schema_version: int = SCHEMA_VERSION
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": self.summary,
            "summarised_through": self.summarised_through,
            "schema_version": self.schema_version,
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        return cls(
            session_id=str(data.get("session_id", "")),
            session_name=str(data.get("session_name", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            turns=[Turn.from_dict(t) for t in data.get("turns") or []],
            summary=str(data.get("summary", "")),
            summarised_through=int(data.get("summarised_through", 0) or 0),
        )

    def add(self, turn: Turn) -> Turn:
        self.turns.append(turn)
        self.updated_at = _now()
        return turn

    @property
    def total_trials(self) -> int:
        """Evaluations spent across the whole thread — what promotion must fold into the session."""
        return sum(
            1
            for turn in self.turns
            for call in turn.calls
            if call.tool == "evaluate_expression" and call.ok
        )

    def proposal(self, proposal_id: str) -> tuple[int, dict[str, Any]] | None:
        """Find a staged proposal and the index of the turn that produced it."""
        for index, turn in enumerate(self.turns):
            for item in turn.proposals:
                if item.get("id") == proposal_id:
                    return index, item
        return None

    def summary_view(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "turns": len(self.turns),
            "updated_at": self.updated_at,
            "trials": self.total_trials,
            "pending_proposals": sum(
                1 for turn in self.turns for item in turn.proposals if not item.get("applied")
            ),
        }


class ConversationStore:
    """Reads and writes the one conversation file that belongs to a session."""

    def get(self, session_id: str) -> Conversation | None:
        path = conversation_path(session_id)
        if not path.exists():
            return None
        try:
            return Conversation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def load_or_create(self, session_id: str, session_name: str = "") -> Conversation:
        existing = self.get(session_id)
        if existing is not None:
            if session_name and existing.session_name != session_name:
                existing.session_name = session_name  # the session was renamed; follow it
            return existing
        return Conversation(session_id=session_id, session_name=session_name)

    def save(self, conversation: Conversation) -> Conversation:
        path = conversation_path(conversation.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(conversation.to_dict(), indent=2, sort_keys=True))
        return conversation

    def delete(self, session_id: str) -> bool:
        path = conversation_path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list(self) -> list[dict[str, Any]]:
        """Every conversation in the workspace, newest first."""
        root = paths.sessions_dir()
        if not root.exists():
            return []
        out: list[dict[str, Any]] = []
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            conversation = self.get(directory.name)
            if conversation is not None and conversation.turns:
                out.append(conversation.summary_view())
        return sorted(out, key=lambda item: item["updated_at"], reverse=True)
