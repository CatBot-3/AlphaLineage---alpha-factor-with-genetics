"""P11-T6 - the conversational agent surface, end to end.

Three properties carry the phase:

* one door — the same message endpoint answers "what does this measure" and "improve this",
  with the model choosing tools rather than the user pre-classifying the question;
* deleting a training session takes its conversation with it, with no second cleanup path;
* an agent round is a real round, but its search cost is folded into the session *before* the
  round exists, because a round is finalizable the moment it does.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from alphalineage.agent import service as agent_service
from alphalineage.agent.conversation import ConversationStore, conversation_path
from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.panel import Panel

TRAIN_END = "2016-01-01"
VALID_START = "2016-01-11"
TEST_START = "2020-01-11"


def make_panel(periods: int = 3400) -> Panel:
    dates = pd.date_range("2010-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(23)
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


@pytest.fixture
def client():
    panel = make_panel()
    app.dependency_overrides[get_panel] = lambda: panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clean_operators():
    yield
    extensions.clear_user_operators()


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "ALPHALINEAGE_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


BEST_FACTOR = {
    "name": "rank",
    "children": [
        {"name": "ts_std", "children": [{"name": "returns"}, {"name": "window", "value": 20}]}
    ],
}


@pytest.fixture
def session_on_disk(temp_data_dir: Path) -> str:
    root = temp_data_dir / "sessions" / "session-agent"
    (root / "rounds").mkdir(parents=True, exist_ok=True)
    (root / "session.json").write_text(
        json.dumps(
            {
                "id": "session-agent",
                "name": "Volatility hunt",
                "universe": "sp500-lite",
                "as_of": "2026-01-05",
                "created_at": "2026-01-05T00:00:00+00:00",
                "boundaries": {
                    "train_end": TRAIN_END,
                    "valid_start": VALID_START,
                    "valid_end": "2020-01-01",
                    "test_start": TEST_START,
                    "embargo": 5,
                },
                "config": {
                    "horizon": 1,
                    "population_size": 200,
                    "generations": 20,
                    "min_names": 5,
                    "max_nodes": 40,
                    "seed": 0,
                    "ic_method": "spearman",
                    "validation_folds": 2,
                },
                "cumulative_trials": 2773,
                "test_reads": 0,
                "session_holdout_reads": 0,
                "operators": [],
                "rounds": [{"index": 0, "status": "done", "evidence_status": "validation_only"}],
            }
        ),
        encoding="utf-8",
    )
    (root / "rounds" / "0000.json").write_text(
        json.dumps(
            {
                "round_index": 0,
                "best_factor": json.dumps(BEST_FACTOR),
                "generations": 20,
                "termination_reason": "completed",
                "evidence_status": "validation_only",
                "cumulative_trials": 2773,
                "test_reads": 0,
                "session_holdout_reads": 0,
                "report": None,
                "context": {"universe": "sp500-lite", "horizon": 1},
                "selection": {
                    "training_metrics": {"ic": 0.017},
                    "validation_metrics": {"oriented_ic": 0.053},
                },
                "history": [{"generation": 20, "best_fitness": 0.17}],
            }
        ),
        encoding="utf-8",
    )
    return "session-agent"


class ScriptedModel:
    def __init__(self, *turns: AIMessage) -> None:
        self.turns, self.calls = list(turns), 0
        self.bound: list[dict[str, Any]] = []
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]) -> ScriptedModel:
        self.bound = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.seen.append(list(messages))
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn


def ai(text: str = "", tool_calls: list[dict[str, Any]] | None = None) -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": c["name"], "args": c.get("args", {}), "id": c.get("id", f"c{i}")}
            for i, c in enumerate(tool_calls or [])
        ],
        usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
    )


def _poll(client: TestClient, job_id: str, timeout: float = 40.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/agent/jobs/{job_id}").json()
        if payload["status"] in ("done", "failed", "stopped"):
            return payload
        time.sleep(0.05)
    return payload


def _send(client: TestClient, session_id: str, monkeypatch, model: ScriptedModel, text: str):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-1234")
    monkeypatch.setattr(agent_service, "build_chat_model", lambda *a, **k: model)
    job = client.post(f"/agent/conversations/{session_id}/messages", json={"message": text}).json()
    payload = _poll(client, job["job_id"])
    # Flatten so a test can read both the job status and the turn it produced.
    return {**payload, **(payload.get("result") or {})}


# --- the tool catalog -------------------------------------------------------------
def test_there_is_one_toolset_not_a_per_mode_split(client: TestClient) -> None:
    payload = client.get("/agent/tools").json()
    names = {tool["name"] for tool in payload["tools"]}
    assert {"annotate_factor", "search_workspace", "list_agent_rounds"} <= names
    assert "toolsets" not in payload  # the per-kind split is gone
    assert {tool["safety"] for tool in payload["tools"]} == {
        "read",
        "evaluate",
        "propose",
        "annotate",
    }


def test_only_one_tool_can_spend_evidence(client: TestClient) -> None:
    payload = client.get("/agent/tools").json()
    spenders = [t["name"] for t in payload["tools"] if t["safety"] == "evaluate"]
    assert spenders == ["evaluate_expression"]


def test_no_tool_accepts_a_date_or_split_argument(client: TestClient) -> None:
    forbidden = {"start", "end", "date", "dates", "split", "universe", "window_start"}
    for tool in client.get("/agent/tools").json()["tools"]:
        assert not forbidden & set(tool["parameters"].get("properties", {}))


# --- conversation lifecycle -------------------------------------------------------
def test_a_fresh_session_has_an_empty_thread_named_after_it(
    client: TestClient, session_on_disk: str
) -> None:
    payload = client.get(f"/agent/conversations/{session_on_disk}").json()
    assert payload["session_name"] == "Volatility hunt"
    assert payload["turns"] == []


def test_sending_a_message_appends_both_turns(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    model = ScriptedModel(
        ai(tool_calls=[{"name": "get_current_factor"}]),
        ai("It measures 20-day return dispersion, ranked across the universe."),
    )
    result = _send(client, session_on_disk, monkeypatch, model, "what does this factor measure?")
    assert result["status"] == "done", result

    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    assert [t["role"] for t in thread["turns"]] == ["user", "assistant"]
    assert thread["turns"][0]["text"] == "what does this factor measure?"
    assert "dispersion" in thread["turns"][1]["text"]
    assert thread["turns"][1]["calls"][0]["tool"] == "get_current_factor"


def test_the_same_endpoint_serves_explain_and_improve(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    """One door: the model picks the tools, the user does not pre-classify the question."""
    explain = ScriptedModel(
        ai(tool_calls=[{"name": "get_current_factor"}]), ai("It reads dispersion.")
    )
    _send(client, session_on_disk, monkeypatch, explain, "explain this")

    improve = ScriptedModel(
        ai(
            tool_calls=[
                {"name": "evaluate_expression", "args": {"expression": "rank(ts_std(returns, 40))"}}
            ]
        ),
        ai("The longer window held up slightly better."),
    )
    _send(client, session_on_disk, monkeypatch, improve, "improve this")

    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    tools_used = [c["tool"] for t in thread["turns"] for c in t["calls"]]
    assert "get_current_factor" in tools_used and "evaluate_expression" in tools_used


def test_history_is_replayed_on_the_next_message(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    first = ScriptedModel(ai("Noted."))
    _send(client, session_on_disk, monkeypatch, first, "remember: I care about turnover")

    second = ScriptedModel(ai("As you mentioned, turnover matters here."))
    _send(client, session_on_disk, monkeypatch, second, "and now?")
    replayed = "\n".join(str(getattr(m, "content", "")) for turn in second.seen for m in turn)
    assert "turnover" in replayed


def test_labels_come_from_the_model_not_a_fixed_list(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    """The hand-written taxonomy is gone; the vocabulary is the model's to choose."""
    model = ScriptedModel(
        ai(
            tool_calls=[
                {
                    "name": "annotate_factor",
                    "args": {
                        "labels": [
                            {
                                "name": "dispersion of recent returns",
                                "evidence": "ts_std(returns, 20)",
                                "reading": "size of variation, not direction",
                                "risk": "low-vol crowding",
                            }
                        ]
                    },
                }
            ]
        ),
        ai("Labelled."),
    )
    _send(client, session_on_disk, monkeypatch, model, "what does it measure?")
    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    labels = thread["turns"][1]["labels"]
    assert labels[0]["name"] == "dispersion of recent returns"


def test_a_message_without_a_key_is_refused_before_any_work(
    client: TestClient, session_on_disk: str
) -> None:
    response = client.post(
        f"/agent/conversations/{session_on_disk}/messages", json={"message": "hi"}
    )
    assert response.status_code == 400
    assert "no API key configured" in response.json()["detail"]


def test_an_unknown_session_is_a_404(client: TestClient) -> None:
    assert client.get("/agent/conversations/nope").status_code == 404
    assert (
        client.post("/agent/conversations/nope/messages", json={"message": "hi"}).status_code == 404
    )


def test_clearing_the_thread_leaves_the_session_alone(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    _send(client, session_on_disk, monkeypatch, ScriptedModel(ai("ok")), "hello")
    assert client.delete(f"/agent/conversations/{session_on_disk}").json()["cleared"] is True
    assert client.get(f"/agent/conversations/{session_on_disk}").json()["turns"] == []
    assert client.get(f"/sessions/{session_on_disk}").status_code == 200


# --- deletion binds the two together ----------------------------------------------
def test_deleting_the_session_deletes_the_conversation(
    client: TestClient, session_on_disk: str, monkeypatch, temp_data_dir: Path
) -> None:
    """The binding requirement: one removal, no second cleanup path to forget."""
    _send(client, session_on_disk, monkeypatch, ScriptedModel(ai("ok")), "hello")
    assert conversation_path(session_on_disk).exists()

    response = client.delete(f"/sessions/{session_on_disk}")
    assert response.status_code == 200
    assert response.json()["conversation_deleted"] is True

    assert not (temp_data_dir / "sessions" / session_on_disk).exists()
    assert not conversation_path(session_on_disk).exists()
    assert ConversationStore().get(session_on_disk) is None
    assert client.get(f"/sessions/{session_on_disk}").status_code == 404


def test_deleting_an_unknown_session_is_a_404(client: TestClient) -> None:
    assert client.delete("/sessions/nope").status_code == 404


# --- promotion --------------------------------------------------------------------
PROPOSE = (
    ai(
        tool_calls=[
            {
                "name": "propose_factor",
                "args": {
                    "expression": "rank(ts_std(returns, 40))",
                    "name": "slower volatility",
                    "rationale": "the 20-day window looked noisy on the inner holdout",
                },
            }
        ]
    ),
    ai("Staged one candidate."),
)


def test_a_staged_factor_changes_nothing_until_promoted(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    _send(client, session_on_disk, monkeypatch, ScriptedModel(*PROPOSE), "improve this")
    rounds = client.get(f"/sessions/{session_on_disk}/rounds").json()
    assert len(rounds) == 1  # still only the GP round


def test_promotion_validates_and_creates_a_real_round(
    client: TestClient, session_on_disk: str, monkeypatch, temp_data_dir: Path
) -> None:
    _send(client, session_on_disk, monkeypatch, ScriptedModel(*PROPOSE), "improve this")
    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    proposal_id = thread["turns"][1]["proposals"][0]["id"]

    job = client.post(
        f"/agent/conversations/{session_on_disk}/proposals/{proposal_id}/promote"
    ).json()
    payload = _poll(client, job["job_id"])
    assert payload["status"] == "done", payload

    rounds = client.get(f"/sessions/{session_on_disk}/rounds").json()
    assert len(rounds) == 2
    agent_round = rounds[-1]
    assert agent_round["origin"] == "agent"
    assert agent_round["parent_round_index"] == 0

    # It carries genuine validation metrics, not just the inner-holdout number the agent saw.
    stored = json.loads(
        (temp_data_dir / "sessions" / session_on_disk / "rounds" / "0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["selection"]["validation_metrics"]
    assert stored["round_metadata"]["origin"] == "agent"
    assert stored["report"] is None  # finalization is still a separate, explicit act


def test_promotion_folds_trials_in_before_the_round_exists(
    client: TestClient, session_on_disk: str, monkeypatch, temp_data_dir: Path
) -> None:
    """A round is finalizable the moment it exists, so its search cost must already be counted."""
    turns = (
        ai(
            tool_calls=[
                {"name": "evaluate_expression", "args": {"expression": "rank(ts_std(returns, 40))"}}
            ]
        ),
        *PROPOSE,
    )
    _send(client, session_on_disk, monkeypatch, ScriptedModel(*turns), "improve this")
    session_path = temp_data_dir / "sessions" / session_on_disk / "session.json"
    assert json.loads(session_path.read_text(encoding="utf-8"))["cumulative_trials"] == 2773

    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    proposal_id = thread["turns"][1]["proposals"][0]["id"]
    job = client.post(
        f"/agent/conversations/{session_on_disk}/proposals/{proposal_id}/promote"
    ).json()
    _poll(client, job["job_id"])

    state = json.loads(session_path.read_text(encoding="utf-8"))
    assert state["cumulative_trials"] == 2774
    stored = json.loads(
        (temp_data_dir / "sessions" / session_on_disk / "rounds" / "0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["cumulative_trials"] == 2774


def test_a_promoted_round_is_visible_to_the_model_next_turn(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    _send(client, session_on_disk, monkeypatch, ScriptedModel(*PROPOSE), "improve this")
    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    proposal_id = thread["turns"][1]["proposals"][0]["id"]
    _poll(
        client,
        client.post(
            f"/agent/conversations/{session_on_disk}/proposals/{proposal_id}/promote"
        ).json()["job_id"],
    )

    model = ScriptedModel(ai(tool_calls=[{"name": "list_agent_rounds"}]), ai("Building on A1."))
    result = _send(client, session_on_disk, monkeypatch, model, "keep going")
    rows = result["turn"]["calls"][0]["result"]["rounds"]
    assert len(rows) == 1 and rows[0]["parent_round_index"] == 0


def test_a_proposal_cannot_be_promoted_twice(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    _send(client, session_on_disk, monkeypatch, ScriptedModel(*PROPOSE), "improve this")
    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    proposal_id = thread["turns"][1]["proposals"][0]["id"]
    path = f"/agent/conversations/{session_on_disk}/proposals/{proposal_id}/promote"
    _poll(client, client.post(path).json()["job_id"])
    assert client.post(path).status_code == 400


def test_an_unknown_proposal_is_a_400(client: TestClient, session_on_disk: str) -> None:
    assert (
        client.post(f"/agent/conversations/{session_on_disk}/proposals/nope/promote").status_code
        == 400
    )


# --- config patches ---------------------------------------------------------------
def test_applying_a_config_patch_updates_the_session(
    client: TestClient, session_on_disk: str, monkeypatch, temp_data_dir: Path
) -> None:
    model = ScriptedModel(
        ai(
            tool_calls=[
                {
                    "name": "propose_config_patch",
                    "args": {
                        "patch": {"generations": 40},
                        "rationale": "fitness was still climbing at the last generation",
                    },
                }
            ]
        ),
        ai("Staged a config change."),
    )
    _send(client, session_on_disk, monkeypatch, model, "how should I tune the next run?")
    thread = client.get(f"/agent/conversations/{session_on_disk}").json()
    proposal_id = thread["turns"][1]["proposals"][0]["id"]

    applied = client.post(
        f"/agent/conversations/{session_on_disk}/proposals/{proposal_id}/apply-config"
    ).json()
    assert applied["diff"] == [{"key": "generations", "from": 20, "to": 40}]
    state = json.loads(
        (temp_data_dir / "sessions" / session_on_disk / "session.json").read_text(encoding="utf-8")
    )
    assert state["config"]["generations"] == 40
    assert state["config"]["seed"] == 0  # protected keys untouched


def test_a_protected_key_is_refused_with_the_reason(
    client: TestClient, session_on_disk: str, monkeypatch
) -> None:
    model = ScriptedModel(
        ai(
            tool_calls=[
                {
                    "name": "propose_config_patch",
                    "args": {"patch": {"seed": 7}, "rationale": "try a luckier run"},
                }
            ]
        ),
        ai("That was refused."),
    )
    result = _send(client, session_on_disk, monkeypatch, model, "tune it")
    assert "seed-hacking" in result["turn"]["calls"][0]["result"]["error"]


# --- the invariant ----------------------------------------------------------------
def test_a_conversation_never_touches_the_locked_splits(
    client: TestClient, session_on_disk: str, monkeypatch, temp_data_dir: Path
) -> None:
    model = ScriptedModel(
        ai(tool_calls=[{"name": "get_current_factor"}]),
        ai(
            tool_calls=[
                {"name": "evaluate_expression", "args": {"expression": "rank(ts_std(returns, 40))"}}
            ]
        ),
        ai("Measured one candidate."),
    )
    result = _send(client, session_on_disk, monkeypatch, model, "improve this")
    assert result["status"] == "done", result

    state = json.loads(
        (temp_data_dir / "sessions" / session_on_disk / "session.json").read_text(encoding="utf-8")
    )
    assert state["test_reads"] == 0
    assert state["session_holdout_reads"] == 0
    assert state["cumulative_trials"] == 2773  # not folded until promotion

    holdout_end = result["turn"]["calls"][1]["result"]["split"]["inner_holdout_end"]
    assert pd.Timestamp(holdout_end) <= pd.Timestamp(TRAIN_END)
    assert pd.Timestamp(holdout_end) < pd.Timestamp(VALID_START)
