"""The universe download script resumes from cache and stops cleanly at a provider allowance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from alphalineage.data import paths, schema
from alphalineage.data.cache import ParquetCache
from alphalineage.data.provider import QuotaExceededError

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_universe.py"


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location("download_universe_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    monkeypatch.setenv("TIINGO_API_KEY", "test-key")
    return module


def test_group_expansion_covers_every_bundled_subset(script) -> None:
    sectors = script.group_universes("sectors")
    themes = script.group_universes("themes")
    assert len(sectors) == 11 and len(themes) == 20
    assert script.group_universes("subsets") == sectors + themes
    pairs = script.ordered_symbols([script.resolve_universe(name) for name in sectors])
    assert len(pairs) == 503
    assert ("BRK-B", "BRK-B") in pairs


def test_quota_stop_spends_no_further_requests_and_resumes_from_cache(
    script, monkeypatch, synthetic_prices, capsys
) -> None:
    prices = schema.normalize(synthetic_prices)
    ParquetCache().store("XOM", prices)  # already cached: must not be requested
    calls: list[str] = []
    quota_on_call = {"n": 3}

    class Provider:
        name = "tiingo"

        def get_prices(self, symbol, start=None, end=None):
            calls.append(symbol)
            if len(calls) == quota_on_call["n"]:
                raise QuotaExceededError("hourly allowance used", scope="hourly", provider="tiingo")
            return prices

    monkeypatch.setattr(script, "_build_provider", lambda *_a, **_k: Provider())
    code = script.main(["--universe", "sp500-energy", "--years", "1"])

    assert code == script.EXIT_QUOTA
    assert "XOM" not in calls and len(calls) == 3
    log = json.loads((paths.meta_dir() / "fetch_log.json").read_text(encoding="utf-8"))
    assert log["quota"]["scope"] == "hourly"
    assert len(log["fetched"]) == 2
    assert calls[2] in log["remaining"]
    assert "allowance exhausted" in capsys.readouterr().out

    # A second run skips everything already cached and continues with the first missing name.
    calls.clear()
    quota_on_call["n"] = -1
    assert script.main(["--universe", "sp500-energy", "--years", "1", "--max-requests", "1"]) == 0
    assert calls == [log["remaining"][0]]
