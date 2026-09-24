"""P10 - the agent must never be a prerequisite for the application starting.

This exists because it regressed once. LangGraph was imported at module scope in ``agent.graph``,
``agent.service`` imported that at module scope, and ``api.app`` imported *that* at module scope —
so a venv created before Phase 10 added the dependency could not import the app at all. uvicorn
imports the app before binding a port, so the process exited instantly and the launcher window
closed with nothing to read.

The rule these tests hold: LangChain and LangGraph are core dependencies, but a missing or broken
install degrades the *agent* to unavailable. Training, explanation, backtesting, and the whole
rest of the API keep working.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

#: Every distribution the agent loop pulls in. Blocking all of them simulates the venv of a user
#: who installed the project before Phase 10 landed.
AGENT_PACKAGES = (
    "langgraph",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langchain_anthropic",
)

#: Modules that must survive the blockade, re-imported inside it.
APP_MODULES = (
    "alphalineage.agent.context",
    "alphalineage.agent.guards",
    "alphalineage.agent.evaluate",
    "alphalineage.agent.expressions",
    "alphalineage.agent.tools",
    "alphalineage.agent.runtime",
    "alphalineage.agent.conversation",
    "alphalineage.agent.memory",
    "alphalineage.agent.vocabulary",
    "alphalineage.agent.promote",
    "alphalineage.agent.service",
    "alphalineage.api.app",
)


@contextmanager
def langchain_missing() -> Iterator[None]:
    """Make every agent dependency un-importable, and force affected modules to reload."""
    real_import = builtins.__import__
    saved = {name: sys.modules.pop(name, None) for name in list(sys.modules) if _agent_owned(name)}

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in AGENT_PACKAGES:
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = real_import
        for name in list(sys.modules):
            if _agent_owned(name):
                del sys.modules[name]
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
        # Restoring ``sys.modules`` is not enough. Importing a submodule also rebinds it as an
        # attribute of its parent package, so the throwaway imports above left
        # ``alphalineage.api.sessions`` (the attribute) pointing at a module nothing else uses
        # while ``sys.modules`` pointed at the real one. ``monkeypatch.setattr`` resolves a
        # dotted target by attribute traversal, so a later test patching
        # "alphalineage.api.sessions.build_report" patched the orphan and the real function
        # went on running - which is how a cancelled finalization published a real artifact.
        for name, module in saved.items():
            if module is None or "." not in name:
                continue
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child, module)


def _agent_owned(name: str) -> bool:
    root = name.split(".")[0]
    return root in AGENT_PACKAGES or name.startswith(("alphalineage.agent", "alphalineage.api"))


# --- the regression --------------------------------------------------------------
@pytest.mark.parametrize("module", APP_MODULES)
def test_every_module_imports_without_the_agent_dependencies(module: str) -> None:
    with langchain_missing():
        importlib.import_module(module)


def test_the_api_serves_normally_without_the_agent_dependencies() -> None:
    """The exact thing uvicorn does, in the exact environment that used to break it."""
    with langchain_missing():
        app_module = importlib.import_module("alphalineage.api.app")
        with TestClient(app_module.app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/primitives").status_code == 200
            assert client.get("/agent/tools").status_code == 200
            assert client.get("/settings").status_code == 200


def test_the_agent_reports_itself_unavailable_rather_than_crashing() -> None:
    with langchain_missing():
        app_module = importlib.import_module("alphalineage.api.app")
        with TestClient(app_module.app) as client:
            payload = client.get("/agent/tools").json()
    # The catalog still renders — a user can see what the agent *would* do — but the reason it
    # cannot run is stated, and it names the fix.
    assert payload["tools"]
    reason = payload["unavailable_reason"]
    assert reason
    assert "langchain" in reason.lower() or "langgraph" in reason.lower()


def test_unavailable_reason_names_the_missing_dependency_and_the_fix() -> None:
    with langchain_missing():
        service = importlib.import_module("alphalineage.agent.service")
        reason = service.unavailable_reason()
    assert reason
    assert "reinstall" in reason.lower() or "install" in reason.lower()


def test_unavailable_reason_is_empty_when_the_dependencies_are_present() -> None:
    from alphalineage.agent import service

    assert service.unavailable_reason() == ""


# --- the boundary ----------------------------------------------------------------
def test_even_the_graph_module_imports_without_langgraph() -> None:
    """The seam is at call time, not import time — so nothing anywhere breaks startup."""
    with langchain_missing():
        graph = importlib.import_module("alphalineage.agent.graph")
        assert graph.SYSTEM_PROMPT  # usable for everything but building a graph


def test_building_a_graph_without_langgraph_names_the_fix() -> None:
    with langchain_missing():
        graph = importlib.import_module("alphalineage.agent.graph")
        runtime = importlib.import_module("alphalineage.agent.runtime")
        with pytest.raises(runtime.AgentUnavailable, match="langgraph"):
            graph._require_langgraph()


def test_starting_a_run_without_the_dependencies_fails_cleanly() -> None:
    """A 4xx-shaped error the UI can show, not an ImportError escaping mid-run."""
    with langchain_missing():
        service = importlib.import_module("alphalineage.agent.service")
        runtime = importlib.import_module("alphalineage.agent.runtime")
        with pytest.raises(runtime.AgentUnavailable):
            runtime._require_langchain()
        assert service.unavailable_reason()
