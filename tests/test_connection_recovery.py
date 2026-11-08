"""Connection failures must retain recoverable state and never claim a new training job."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import alphalineage.api.app as api_app
from alphalineage.api import launcher, resources, sessions
from alphalineage.api.errors import ActionError
from alphalineage.api.jobs import JobStore
from alphalineage.api.progress import RunProgress
from alphalineage.core.gp import EVOLUTION_VERSION, GP, GPConfig
from alphalineage.core.tree import Node, to_dict
from alphalineage.data.identifiers import atomic_write_text


@pytest.fixture
def client(signal_panel, monkeypatch):
    monkeypatch.setattr(api_app, "_jobs", JobStore())
    api_app.app.dependency_overrides[api_app.get_panel] = lambda: signal_panel[0]
    with TestClient(api_app.app) as http:
        yield http
    api_app.app.dependency_overrides.clear()


def _create(client):
    response = client.post("/sessions", json={
        "name": "recovery", "as_of": "2020-01-01",
        "config": {"population_size": 12, "generations": 1,
                   "max_depth": 3, "max_nodes": 12, "seed": 11, "novelty_mode": "off"},
    })
    assert response.status_code == 200, response.text
    handle = response.json()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = client.get(f"/runs/{handle['job_id']}").json()
        if job["status"] != "running" and job["status"] != "queued":
            assert job["status"] == "done", job
            return handle["session_id"]
        time.sleep(0.02)
    pytest.fail("synthetic run did not finish")


@pytest.mark.parametrize("physical,commit", [(1 << 30, 8 << 30), (8 << 30, 1 << 30)])
@pytest.mark.parametrize("endpoint", ["/runs", "/sessions", "/sessions/missing/continue"])
def test_low_memory_rejects_before_panel_load_or_job_claim(client, monkeypatch, physical,
                                                         commit, endpoint):
    monkeypatch.setattr(resources, "memory_headroom_bytes", lambda: (physical, commit))
    def forbidden_panel():
        pytest.fail("low-memory request loaded market data")
    api_app.app.dependency_overrides[api_app.get_panel] = forbidden_panel
    response = client.post(endpoint, json={"as_of": "2020-01-01", "generations": 1})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "low_training_memory"
    assert response.json()["detail"]["retryable"] is True
    assert api_app._jobs.list() == []
    assert not sessions.paths.sessions_dir().exists()


def test_headroom_thresholds_and_unavailable_commit_detection(monkeypatch):
    for headroom in [(2 << 30, 2 << 30), (2 << 30, None)]:
        monkeypatch.setattr(resources, "memory_headroom_bytes", lambda h=headroom: h)
        resources.ensure_training_memory()
    monkeypatch.setattr(resources, "memory_headroom_bytes", lambda: ((2 << 30) - 1, None))
    with pytest.raises(ActionError, match="saved checkpoint is unchanged"):
        resources.ensure_training_memory()


def test_summary_caches_only_scalars_and_invalidates_on_atomic_replace(monkeypatch):
    path = sessions.session_dir("legacy") / "checkpoint.json"
    path.parent.mkdir(parents=True)
    payload = {"generation": 15, "trials": 21916, "config": {"generations": 50},
               "evolution_version": EVOLUTION_VERSION, "population": ["bulk" * 1024]}
    atomic_write_text(path, json.dumps(payload))
    original = sessions._read_result
    reads = []
    def read(file):
        reads.append(file)
        return original(file)
    monkeypatch.setattr(sessions, "_read_result", read)
    first = sessions.checkpoint_summary("legacy")
    first["generation"] = 999  # callers cannot alter cached data
    assert sessions.checkpoint_summary("legacy")["generation"] == 15
    assert len(reads) == 1
    assert set(sessions._checkpoint_summaries[str(path.resolve())][1]) == {
        "generation", "trials", "target_generations", "evolution_version"
    }
    payload["generation"] = 16
    atomic_write_text(path, json.dumps(payload))
    assert sessions.checkpoint_summary("legacy")["generation"] == 16
    assert len(reads) == 2


def test_new_snapshot_header_does_not_decode_bulk_payload(monkeypatch):
    path = sessions.session_dir("header") / "checkpoint.json"
    path.parent.mkdir(parents=True)
    summary = {"generation": 15, "trials": 21916, "target_generations": 50,
               "evolution_version": EVOLUTION_VERSION}
    atomic_write_text(path, json.dumps({"recovery_summary": summary,
                                       "population": ["x" * 1024**2]}))
    def forbidden_read(_):
        pytest.fail("status polling decoded the full checkpoint")
    monkeypatch.setattr(sessions, "_read_result", forbidden_read)
    assert sessions.checkpoint_summary("header") == summary


def test_restarted_app_exposes_checkpoint_and_preserves_intended_target(client):
    session_id = _create(client)
    assert sessions.claim_job(session_id, "lost-process", requested_generations=49)
    state = client.get(f"/sessions/{session_id}").json()
    assert state["job"]["status"] == "interrupted"
    assert state["job"]["progress"]["generation"] == 1
    assert state["job"]["progress"]["target_generations"] == 50
    assert state["recovery"]["can_resume"] is True
    assert state["cumulative_trials"] == state["recovery"]["trials"]
    assert sessions.load_session(session_id)["active_job_id"] is None


def test_continue_reads_full_snapshot_once_and_seeds_reference_progress(client, monkeypatch):
    session_id = _create(client)
    original_read = sessions._read_result
    checkpoint_reads = []
    observed = []
    original_restore = GP.from_checkpoint_state.__func__
    def read(path):
        if path.name == "checkpoint.json":
            checkpoint_reads.append(path)
        return original_read(path)
    def restore(cls, state, panel, **kwargs):
        progress = kwargs["recorder"]
        progress.on_reference_preparation(0, 20)
        observed.append(progress.snapshot())
        return original_restore(cls, state, panel, **kwargs)
    monkeypatch.setattr(sessions, "_read_result", read)
    monkeypatch.setattr(GP, "from_checkpoint_state", classmethod(restore))
    handle = client.post(f"/sessions/{session_id}/continue", json={"generations": 1})
    assert handle.status_code == 200
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = client.get(f"/runs/{handle.json()['job_id']}").json()
        if job["status"] not in ("running", "queued"):
            break
        time.sleep(0.02)
    assert job["status"] == "done", job
    assert len(checkpoint_reads) == 1
    assert observed[0]["generation"] == 1
    assert observed[0]["target_generations"] == 2
    assert observed[0]["phase"] == "preparing_references"


def test_job_failure_keeps_original_error_and_logs_traceback(caplog):
    store = JobStore()
    def fail():
        raise RuntimeError("original training failure")
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        job_id = store.submit(fail)
        deadline = time.monotonic() + 3
        while store.get(job_id).status != "failed" and time.monotonic() < deadline:
            time.sleep(0.01)
    assert store.get(job_id).error == "original training failure"
    assert "Traceback" in caplog.text
    assert job_id in caplog.text


@pytest.mark.parametrize("owned", [True, False])
def test_backend_records_shutdown_only_for_its_own_launch(tmp_path, monkeypatch, owned):
    path = tmp_path / "server.json"
    state = {"launch_id": "owned" if owned else "another-launch",
             "backend_pid": os.getpid(), "status": "starting"}
    atomic_write_text(path, json.dumps(state))
    monkeypatch.setenv("ALPHALINEAGE_SERVER_STATE", str(path))
    monkeypatch.setenv("ALPHALINEAGE_INSTANCE_ID", "owned")
    launcher.record_shutdown("API shutdown", 0)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if owned:
        assert saved["status"] == "stopped"
        assert saved["shutdown_reason"] == "API shutdown"
        assert saved["exit_code"] == 0
        assert saved["stopped_at"]
    else:
        assert saved == state


def test_shutdown_diagnostics_write_failure_does_not_prevent_exit(tmp_path, monkeypatch,
                                                                caplog):
    path = tmp_path / "server.json"
    original = json.dumps({"launch_id": "owned", "backend_pid": os.getpid()})
    atomic_write_text(path, original)
    monkeypatch.setenv("ALPHALINEAGE_SERVER_STATE", str(path))
    monkeypatch.setenv("ALPHALINEAGE_INSTANCE_ID", "owned")

    def fail_write(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(launcher, "atomic_write_text", fail_write)
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        launcher.record_shutdown("API shutdown", 0)
    assert path.read_text(encoding="utf-8") == original
    assert "Cannot record backend shutdown" in caplog.text


def test_restored_progress_does_not_record_duplicate_lineage():
    progress = RunProgress()
    progress.restore_checkpoint({"generation": 15, "history": []}, target_generations=50)
    progress.on_reference_preparation(1133, 21917)
    assert progress.snapshot()["generation"] == 15
    assert progress.snapshot()["target_generations"] == 50


def test_continue_acknowledges_before_slow_data_preparation_and_prevents_duplicate(client,
                                                                                monkeypatch):
    session_id = _create(client)
    entered, release = threading.Event(), threading.Event()
    original = api_app._panel_for_universe

    def blocked_panel(*args, **kwargs):
        entered.set()
        assert release.wait(10), "data preparation was never released"
        return original(*args, **kwargs)

    monkeypatch.setattr(api_app, "_panel_for_universe", blocked_panel)
    with ThreadPoolExecutor(max_workers=1) as requests:
        submission = requests.submit(client.post, f"/sessions/{session_id}/continue",
                                     json={"generations": 25})
        try:
            assert entered.wait(3)
            response = submission.result(timeout=2)
            assert response.status_code == 200, response.text
            state = client.get(f"/sessions/{session_id}").json()
            assert state["job"]["status"] == "running"
            assert state["job"]["progress"]["phase"] == "preparing_data"
            assert state["job"]["progress"]["generation"] == 1
            assert state["job"]["progress"]["target_generations"] == 26
            duplicate = client.post(f"/sessions/{session_id}/continue", json={"generations": 25})
            assert duplicate.status_code == 409
            assert client.post(f"/sessions/{session_id}/stop").json()["stopping"]
        finally:
            release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get(f"/runs/{response.json()['job_id']}").json()
        if job["status"] not in ("running", "queued"):
            break
        time.sleep(0.01)
    assert job["status"] == "stopped", job
    assert sessions.checkpoint_summary(session_id)["generation"] == 1


def test_reduced_target_overrides_older_checkpoint_plan(client):
    session_id = _create(client)
    path = sessions.session_dir(session_id) / "checkpoint.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["session_state"]["target_generations"] = 50
    snapshot["recovery_summary"]["target_generations"] = 50
    atomic_write_text(path, json.dumps(snapshot))
    assert sessions.claim_job(session_id, "shorter-run", requested_generations=24)
    state = client.get(f"/sessions/{session_id}").json()
    assert state["recovery"]["generation"] == 1
    assert state["recovery"]["target_generations"] == 25
    assert state["job"]["progress"]["target_generations"] == 25


def test_data_preparation_failure_is_visible_and_retains_checkpoint(client, monkeypatch):
    session_id = _create(client)
    path = sessions.session_dir(session_id) / "checkpoint.json"
    original = path.read_bytes()

    def fail_panel(*args, **kwargs):
        raise ValueError("market data preparation failed")

    monkeypatch.setattr(api_app, "_panel_for_universe", fail_panel)
    response = client.post(f"/sessions/{session_id}/continue", json={"generations": 25})
    assert response.status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get(f"/sessions/{session_id}").json()
        if state["job"]["status"] == "failed":
            break
        time.sleep(0.01)
    assert state["job"]["error"] == "market data preparation failed"
    assert state["active_job_id"] is None
    assert path.read_bytes() == original


def test_stop_during_reference_rebuilding_closes_windows_mappings(signal_panel):
    progress = RunProgress()
    cancel = threading.Event()
    observed = []
    original = progress.on_reference_preparation

    def stop_after_first(done, total):
        observed.append(done)
        if done == 1:
            cancel.set()
        original(done, total)

    progress.on_reference_preparation = stop_after_first
    specs = [{"key": field, "tree": to_dict(Node(field))} for field in ("close", "volume")]

    def rebuild():
        with GP(GPConfig(population_size=12, generations=1, min_names=5,
                         novelty_mode="balanced", max_depth=3, max_nodes=12),
                signal_panel[0], recorder=progress, memory_budget_bytes=4,
                novelty_state={"references": specs}):
            pytest.fail("cancelled reference preparation reached the training engine")

    jobs = JobStore()
    job_id = jobs.submit(rebuild, progress=progress, cancel=cancel)
    deadline = time.monotonic() + 5
    while jobs.get(job_id).status not in ("stopped", "failed") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert jobs.get(job_id).status == "stopped", jobs.get(job_id).error
    assert observed == [0, 1]
    assert not list(sessions.paths.temporary_dir().iterdir())
