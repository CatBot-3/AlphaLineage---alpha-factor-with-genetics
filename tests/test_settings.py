"""L1 acceptance: settings for the Tiingo key and the evaluator backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app
from alphalineage.core import cpp
from alphalineage.data import paths


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    cpp.set_backend(None)  # don't leak the override into other tests


def test_settings_reports_evaluator_and_key_state(client):
    payload = client.get("/settings").json()
    assert "factors_dir" in payload
    assert payload["tiingo_api_key_set"] is False
    assert payload["evaluator"] == "auto"
    assert isinstance(payload["cpp_available"], bool)


def test_set_evaluator_roundtrips_and_applies(client):
    put = client.put("/settings", json={"evaluator": "python"})
    assert put.status_code == 200
    assert put.json()["evaluator"] == "python"
    assert client.get("/settings").json()["evaluator"] == "python"
    assert cpp.selected_backend() == "python"  # the runtime dispatch honors it


def test_invalid_evaluator_rejected(client):
    assert client.put("/settings", json={"evaluator": "gpu"}).status_code == 400


def test_tiingo_key_stored_but_never_echoed(client):
    put = client.put("/settings", json={"tiingo_api_key": "secret-key-123"})
    assert put.status_code == 200
    body = put.json()
    assert body["tiingo_api_key_set"] is True
    assert "secret-key-123" not in str(body)  # the secret is never returned
    assert paths.tiingo_api_key() == "secret-key-123"  # but the backend can resolve it

    # clearing it
    client.put("/settings", json={"tiingo_api_key": ""})
    assert client.get("/settings").json()["tiingo_api_key_set"] is False


def test_partial_update_preserves_other_keys(client):
    client.put("/settings", json={"evaluator": "python"})
    client.put("/settings", json={"tiingo_api_key": "k"})
    settings = client.get("/settings").json()
    assert settings["evaluator"] == "python"  # not wiped by the key update
    assert settings["tiingo_api_key_set"] is True


def test_tiingo_env_overrides_setting(client, monkeypatch):
    client.put("/settings", json={"tiingo_api_key": "from-settings"})
    monkeypatch.setenv("TIINGO_API_KEY", "from-env")
    assert paths.tiingo_api_key() == "from-env"  # env wins (P-S3)
