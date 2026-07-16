from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app, get_panel
from alphalineage.data.cache import ParquetCache
from alphalineage.data.identifiers import child_path, validate_symbol
from alphalineage.library.factors import FactorStore


@pytest.mark.parametrize(
    "value",
    ["../outside", "..\\outside", "C:outside", "/absolute", "NUL", "name. ", ""],
)
def test_storage_children_reject_unsafe_identifiers(tmp_path, value):
    with pytest.raises(ValueError):
        child_path(tmp_path, value, ".json")


@pytest.mark.parametrize("symbol", ["../AAPL", "..\\AAPL", "C:AAPL", "NUL", "AAPL/USD"])
def test_price_cache_rejects_path_like_symbols(tmp_path, symbol):
    cache = ParquetCache(tmp_path)
    with pytest.raises(ValueError):
        cache.path_for(symbol)


def test_market_symbol_normalization_keeps_supported_punctuation(tmp_path):
    assert validate_symbol(" brk-b ") == "BRK-B"
    assert ParquetCache(tmp_path).path_for("brk-b") == tmp_path.resolve() / "BRK-B.parquet"


def test_factor_store_rejects_path_traversal(tmp_path):
    store = FactorStore(tmp_path / "factors")
    with pytest.raises(ValueError):
        store.get("../meta/settings")


def test_factor_api_returns_client_error_for_unsafe_id(synthetic_panel):
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/factors/NUL").status_code == 400
            assert client.patch("/factors/NUL", json={"notes": "x"}).status_code == 400
            assert client.delete("/factors/NUL").status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_data_api_rejects_unsafe_symbols_before_queueing(synthetic_panel):
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/data/coverage", params={"symbols": "NUL"}).status_code == 400
            assert (
                client.post(
                    "/data/sync",
                    json={"symbols": ["NUL"], "start": "2020-01-01"},
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/universes/sync-dates",
                    json={"symbols": ["NUL"], "expected_start": "2020-01-01"},
                ).status_code
                == 400
            )
            validation = client.post("/symbols/validate", json={"symbol": "NUL"})
            assert validation.status_code == 200
            assert validation.json()["valid"] is False
    finally:
        app.dependency_overrides.clear()
