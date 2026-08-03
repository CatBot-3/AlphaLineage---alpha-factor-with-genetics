"""P11-T1 - the conversation store and the three-stage memory compaction.

Two properties matter here and both are about trust rather than mechanics:

* deleting a training session must take its conversation with it, with no second cleanup path;
* compaction must never let a model rewrite a number. Stages 1 and 2 are deterministic; only
  stage 3 involves a model, and it must not fire while stage 2 can still do the work.
"""

from __future__ import annotations

from pathlib import Path

from alphalineage.agent.conversation import (
    ASSISTANT,
    USER,
    Conversation,
    ConversationStore,
    ToolCallRecord,
    Turn,
    conversation_path,
)
from alphalineage.agent.memory import (
    DEFAULT_VERBATIM_TURNS,
    build_messages,
    needs_summary,
    plan_replay,
    render_turn,
    summarise,
)


def evaluate_call(expression: str = "rank(close)", ic: float = 0.041) -> ToolCallRecord:
    return ToolCallRecord(
        tool="evaluate_expression",
        arguments={"expression": expression},
        result={
            "inner_holdout": {"oriented_ic": ic},
            "generalization_gap": 0.012,
            "correlation_with_current_factor": 0.21,
            "padding": "x" * 3000,  # tool results are the bulk of the tokens
        },
        safety="evaluate",
        ok=True,
    )


def thread(turns: int = 8) -> Conversation:
    conversation = Conversation(session_id="s1", session_name="Volatility hunt")
    for index in range(turns):
        conversation.add(Turn(role=USER, text=f"question {index}"))
        conversation.add(
            Turn(
                role=ASSISTANT,
                text=f"answer {index}",
                calls=[evaluate_call(f"rank(ts_std(returns, {index + 5}))", 0.04 + index / 1000)],
            )
        )
    return conversation


# --- binding to a session ---------------------------------------------------------
def test_the_conversation_lives_inside_the_session_directory(temp_data_dir: Path) -> None:
    """This placement is the whole mechanism behind 'delete the session, lose the thread'."""
    path = conversation_path("session-abc")
    assert path.parent.name == "session-abc"
    assert path.parent.parent.name == "sessions"
    assert path.name == "conversation.json"


def test_a_conversation_takes_the_session_name(temp_data_dir: Path) -> None:
    store = ConversationStore()
    conversation = store.load_or_create("session-abc", "Volatility hunt")
    assert conversation.session_name == "Volatility hunt"
    store.save(conversation)
    assert store.get("session-abc").session_name == "Volatility hunt"


def test_renaming_the_session_updates_the_conversation(temp_data_dir: Path) -> None:
    store = ConversationStore()
    store.save(store.load_or_create("session-abc", "Old name"))
    assert store.load_or_create("session-abc", "New name").session_name == "New name"


def test_round_trips_through_disk(temp_data_dir: Path) -> None:
    store = ConversationStore()
    conversation = store.load_or_create("session-abc", "s")
    conversation.add(Turn(role=USER, text="hello"))
    conversation.add(Turn(role=ASSISTANT, text="hi", calls=[evaluate_call()]))
    store.save(conversation)

    loaded = store.get("session-abc")
    assert [turn.text for turn in loaded.turns] == ["hello", "hi"]
    assert loaded.turns[1].calls[0].tool == "evaluate_expression"
    assert loaded.turns[1].calls[0].result["inner_holdout"]["oriented_ic"] == 0.041


def test_a_corrupt_file_reads_as_absent_rather_than_raising(temp_data_dir: Path) -> None:
    path = conversation_path("session-abc")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert ConversationStore().get("session-abc") is None


def test_trials_are_counted_across_the_whole_thread(temp_data_dir: Path) -> None:
    conversation = thread(turns=3)
    assert conversation.total_trials == 3


def test_a_failed_evaluation_is_not_counted_as_a_trial() -> None:
    conversation = Conversation(session_id="s1")
    failed = evaluate_call()
    failed.ok = False
    conversation.add(Turn(role=ASSISTANT, calls=[failed]))
    assert conversation.total_trials == 0


# --- digests are computed, never generated ----------------------------------------
def test_an_evaluation_digest_copies_the_numbers_verbatim() -> None:
    digest = evaluate_call("rank(close)", 0.041).digest()
    assert "0.0410" in digest and "gap 0.0120" in digest and "corr 0.2100" in digest


def test_a_failed_call_digests_to_its_error() -> None:
    call = ToolCallRecord(tool="validate_expression", result={"error": "boom"}, ok=False)
    assert "error" in call.digest() and "boom" in call.digest()


# --- stage 1 and 2 ----------------------------------------------------------------
def test_recent_turns_keep_their_tool_results_in_full() -> None:
    conversation = thread(turns=6)
    blocks, plan = plan_replay(conversation, verbatim_turns=2, budget_chars=200_000)
    assert plan.verbatim_turns == 2
    assert "padding" in blocks[-1]  # the newest assistant turn still carries the whole payload


def test_older_tool_results_collapse_to_a_digest_but_keep_their_numbers() -> None:
    conversation = thread(turns=6)
    blocks, plan = plan_replay(conversation, verbatim_turns=2, budget_chars=200_000)
    old = "\n".join(blocks[:-4])
    assert "padding" not in old  # the bulky payload is gone
    assert "inner-holdout IC" in old  # but the number that mattered survived
    assert plan.digested_turns > 0


def test_digesting_is_what_keeps_a_long_thread_inside_budget() -> None:
    conversation = thread(turns=12)
    _, plan = plan_replay(conversation, verbatim_turns=3, budget_chars=24_000)
    assert plan.characters <= 24_000
    assert plan.verbatim_turns >= 1


def test_the_newest_turn_is_never_the_one_dropped() -> None:
    conversation = thread(turns=20)
    blocks, _ = plan_replay(conversation, verbatim_turns=1, budget_chars=6_000)
    assert "answer 19" in "\n".join(blocks)


def test_one_runaway_tool_result_is_clipped_rather_than_evicting_the_thread() -> None:
    conversation = Conversation(session_id="s1")
    huge = ToolCallRecord(tool="list_operators", result={"blob": "y" * 500_000}, ok=True)
    conversation.add(Turn(role=USER, text="hi"))
    conversation.add(Turn(role=ASSISTANT, text="ok", calls=[huge]))
    blocks, _ = plan_replay(conversation, verbatim_turns=3, budget_chars=200_000)
    assert "kept on disk" in "\n".join(blocks)


def test_render_is_deterministic() -> None:
    turn = thread(turns=1).turns[1]
    assert render_turn(turn, verbatim=False) == render_turn(turn, verbatim=False)


# --- stage 3 ----------------------------------------------------------------------
def test_a_short_thread_never_needs_a_summary() -> None:
    assert needs_summary(thread(turns=2)) is False


def test_summary_only_fires_once_digesting_is_not_enough() -> None:
    assert needs_summary(thread(turns=60), budget_chars=4_000) is True


def test_summarising_folds_the_oldest_turns_and_advances_the_marker() -> None:
    conversation = thread(turns=10)
    calls: list[str] = []

    def fake_model(prompt: str) -> str:
        calls.append(prompt)
        return "Tried five volatility variants; best inner-holdout IC 0.048."

    summarise(conversation, invoke=fake_model, keep_recent=6)
    assert len(calls) == 1
    assert "0.048" in conversation.summary
    assert conversation.summarised_through == len(conversation.turns) - 6


def test_the_summary_prompt_forbids_rewriting_numbers() -> None:
    conversation = thread(turns=10)
    captured: list[str] = []
    summarise(conversation, invoke=lambda p: captured.append(p) or "s", keep_recent=4)
    assert "copy it" in captured[0]
    assert "Do not round" in captured[0]


def test_summarised_turns_are_replaced_by_the_summary_on_replay() -> None:
    conversation = thread(turns=10)
    summarise(conversation, invoke=lambda _: "earlier work", keep_recent=4)
    blocks, plan = plan_replay(conversation, budget_chars=200_000)
    assert plan.summary_used
    assert "earlier work" in blocks[0]
    assert "answer 0" not in "\n".join(blocks)


def test_summarising_twice_does_not_re_summarise_the_same_turns() -> None:
    conversation = thread(turns=10)
    summarise(conversation, invoke=lambda _: "first", keep_recent=6)
    marker = conversation.summarised_through
    summarise(conversation, invoke=lambda _: "second", keep_recent=6)
    assert conversation.summarised_through == marker


# --- message assembly -------------------------------------------------------------
def _factory(role: str, text: str) -> dict[str, str]:
    return {"role": role, "text": text}


def test_messages_carry_system_header_history_and_the_new_message() -> None:
    messages, plan = build_messages(
        thread(turns=2),
        system_prompt="SYSTEM",
        header="HEADER",
        new_message="improve this",
        message_factory=_factory,
    )
    assert messages[0] == {"role": "system", "text": "SYSTEM"}
    assert "HEADER" in messages[1]["text"]
    assert "conversation so far" in messages[1]["text"]
    assert messages[-1]["text"] == "improve this"
    assert plan.estimated_tokens > 0


def test_an_empty_thread_produces_no_history_block() -> None:
    messages, _ = build_messages(
        Conversation(session_id="s1"),
        system_prompt="SYSTEM",
        header="HEADER",
        new_message="hello",
        message_factory=_factory,
    )
    assert "conversation so far" not in messages[1]["text"]


def test_default_verbatim_window_is_small_enough_to_be_useful() -> None:
    assert 1 <= DEFAULT_VERBATIM_TURNS <= 5
