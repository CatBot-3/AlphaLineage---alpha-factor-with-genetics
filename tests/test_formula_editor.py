"""Tests for the Formula Editor backend: dependency-ordered reload, dry-run validate,
formula update (PUT), and GP-mutability of user-formula window arguments."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.primitives import OPERATORS, REGISTRY
from alphalineage.library.indicator_catalog import CATALOG_NAMES


@pytest.fixture(autouse=True)
def _restore_registry():
    ops, reg, users = dict(OPERATORS), dict(REGISTRY), dict(extensions.USER_OPERATORS)
    yield
    for table, saved in ((OPERATORS, ops), (REGISTRY, reg), (extensions.USER_OPERATORS, users)):
        table.clear()
        table.update(saved)


@pytest.fixture
def client(synthetic_panel):
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _write_formulas(payload: list[dict]) -> None:
    from alphalineage.data import paths

    paths.meta_dir().mkdir(parents=True, exist_ok=True)
    paths.formulas_path().write_text(json.dumps(payload), encoding="utf-8")


def test_composed_formula_reloads_regardless_of_file_order(client):
    # `ma_momentum` references `ma`, but is written FIRST in the file. A naive in-order
    # loader would fail to register it; the topo sort must register `ma` first.
    ma = {
        "name": "ma",
        "display_name": "ma",
        "description": "",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    ma_momentum = {
        "name": "ma_momentum",
        "display_name": "ma momentum",
        "description": "",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "sub",
            "children": [
                {
                    "name": "ma",
                    "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
                },
                {"name": "$arg", "value": 0},
            ],
        },
    }
    _write_formulas([ma_momentum, ma])  # dependency listed AFTER its user

    formulas = client.get("/formulas").json()
    by_name = {f["name"]: f for f in formulas}
    assert by_name["ma"]["registered"] is True
    assert by_name["ma_momentum"]["registered"] is True
    assert by_name["ma_momentum"]["error"] is None


def test_validate_reports_type_error_without_registering(client):
    bad = {
        "name": "broken_formula",
        "arg_types": ["series"],
        "out_type": "series",
        "body": {"name": "not_a_primitive"},
    }
    res = client.post("/formulas/validate", json=bad)
    assert res.status_code == 200
    assert res.json()["ok"] is False
    assert "not_a_primitive" in res.json()["error"]
    # the dry run must NOT have registered anything
    assert "broken_formula" not in extensions.USER_OPERATORS

    good = {
        "name": "good_formula",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    ok = client.post("/formulas/validate", json=good)
    assert ok.json()["ok"] is True
    assert ok.json()["out_type"] == "series"
    assert "good_formula" not in extensions.USER_OPERATORS  # still no side effect


def test_put_formula_updates_body_and_dependents_see_it(client):
    base = {
        "name": "dif",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "delta",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    assert client.post("/formulas", json=base).status_code == 200

    # update the body (keep the same signature) -> ts_mean instead of delta
    updated = {
        **base,
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    res = client.put("/formulas/dif", json=updated)
    assert res.status_code == 200
    stored = {f["name"]: f for f in client.get("/formulas").json()}["dif"]
    assert stored["body"]["name"] == "ts_mean"


def test_put_rejects_signature_change_while_in_use(client):
    leaf = {
        "name": "leaf_fn",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    user = {
        "name": "user_fn",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "leaf_fn",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    assert client.post("/formulas", json=leaf).status_code == 200
    assert client.post("/formulas", json=user).status_code == 200

    # changing leaf_fn's arity while user_fn depends on it must be rejected
    bad = {
        **leaf,
        "arg_types": ["series"],
        "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
    }
    res = client.put("/formulas/leaf_fn", json=bad)
    assert res.status_code == 400


def test_primitive_info_includes_category(client):
    prims = {p["name"]: p for p in client.get("/primitives").json()}
    assert prims["close"]["category"] == "data"
    assert prims["ts_mean"]["category"] == "time_series"
    assert prims["rank"]["category"] == "cross_sectional"
    assert prims["add"]["category"] == "arithmetic"


def test_category_resolution_order(client):
    # a user formula declaring its own category
    spec = {
        "name": "my_signal",
        "arg_types": ["series"],
        "out_type": "signal",
        "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
        "category": "custom",
    }
    assert client.post("/formulas", json=spec).status_code == 200
    prims = {p["name"]: p for p in client.get("/primitives").json()}
    assert prims["my_signal"]["category"] == "custom"

    # an override beats both the formula's own category and the built-in default
    assert client.put("/categories/my_signal", json={"category": "alpha"}).status_code == 200
    assert client.put("/categories/close", json={"category": "fields"}).status_code == 400
    prims = {p["name"]: p for p in client.get("/primitives").json()}
    assert prims["my_signal"]["category"] == "alpha"
    assert prims["close"]["category"] == "data"


def test_recategorize_unknown_primitive_is_404(client):
    assert client.put("/categories/not_a_primitive", json={"category": "x"}).status_code == 404


def test_create_and_rename_category(client):
    res = client.put("/categories", json={"order": ["data", "momentum", "custom"]})
    assert res.status_code == 200
    assert "momentum" in res.json()["order"]
    # a brand-new category referenced by an override is auto-appended to the order
    res = client.put("/categories", json={"overrides": {"ts_mean": "trend"}})
    assert "trend" in res.json()["order"]


def test_legacy_category_settings_remove_duplicate_data_groups(client):
    from alphalineage.data import paths

    paths.write_categories(
        {
            "order": ["Data", "data", "constant", "custom"],
            "overrides": {
                "close": "fields",
                "zscore": "data",
                "ts_mean": "trend",
            },
        }
    )
    settings = client.get("/categories").json()
    assert settings["order"].count("data") == 1
    assert "Data" not in settings["order"]
    assert settings["overrides"] == {"ts_mean": "trend"}

    response = client.post(
        "/formulas",
        json={
            "name": "reserved_data_formula",
            "arg_types": [],
            "out_type": "series",
            "body": {"name": "close"},
            "category": "Data",
        },
    )
    assert response.status_code == 200
    assert response.json()["category"] == "custom"


def test_user_formula_window_arg_is_gp_mutable(synthetic_panel):
    # A user formula's WINDOW argument is a real leaf at the call site, so the GP's point
    # mutation re-samples it -> the parameter is genuinely tunable in training (R3c).
    from alphalineage.core.extensions import ARG, register_operator
    from alphalineage.core.gp import GP, GPConfig
    from alphalineage.core.tree import Node
    from alphalineage.core.types import DType

    register_operator(
        "ma_test",
        [DType.SERIES, DType.WINDOW],
        DType.SERIES,
        Node("ts_mean", (Node(ARG, value=0), Node(ARG, value=1))),
    )
    tree = Node("ma_test", (Node("close"), Node("window", value=5)))
    gp = GP(GPConfig(population_size=8, generations=1), synthetic_panel, root_type=DType.SERIES)

    observed: set[int] = set()
    for s in range(300):
        gp.rng.seed(s)
        mutated = gp._point_mutation(tree)
        for node in mutated.iter_nodes():
            if node.name == "window" and node.value is not None:
                observed.add(int(node.value))
    assert observed - {5}, "GP point mutation never altered the call-site window argument"


def test_saved_formula_keeps_parameter_policy_on_initial_registration_and_reload(client):
    spec = {
        "name": "bounded_ma",
        "arg_types": ["series", "window"],
        "inputs": [
            {"name": "series", "type": "series", "role": "data"},
            {
                "name": "lookback",
                "type": "window",
                "role": "parameter",
                "default": 10,
                "tuning": {"min": 5, "max": 20, "step": 1, "radius": 2},
            },
        ],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    response = client.post("/formulas", json=spec)
    assert response.status_code == 200, response.text
    assert REGISTRY["bounded_ma"].macro_policy["inputs"][1]["default"] == 10

    loaded = {item["name"]: item for item in client.get("/formulas").json()}
    assert loaded["bounded_ma"]["registered"] is True
    assert REGISTRY["bounded_ma"].macro_policy["inputs"][1]["tuning"]["radius"] == 2


def test_primitive_catalog_has_readable_metadata_and_named_inputs(client):
    primitives = client.get("/primitives").json()
    assert primitives
    for primitive in primitives:
        assert primitive["display_name"]
        assert primitive["description"]
        assert primitive["origin"] in {
            "builtin",
            "user_formula",
            "catalog_formula",
            "data",
            "value",
        }
        assert len(primitive["inputs"]) == len(primitive["arg_types"])
        assert all(item["name"] and item["type"] for item in primitive["inputs"])
    assert next(item for item in primitives if item["name"] == "ts_mean")["inputs"] == [
        {
            "name": "series",
            "type": "series",
            "description": "Series to average.",
        },
        {
            "name": "lookback",
            "type": "window",
            "description": "Number of periods in the trailing window.",
        },
    ]


def test_packaged_indicator_catalog_is_managed_searchable_and_idempotent(client):
    from alphalineage.data import paths

    first = client.get("/formulas")
    assert first.status_code == 200
    formulas = {item["name"]: item for item in first.json()}
    assert CATALOG_NAMES <= formulas.keys()
    assert all(formulas[name]["origin"] == "catalog_formula" for name in CATALOG_NAMES)
    assert all(formulas[name]["editable"] is False for name in CATALOG_NAMES)
    assert all(formulas[name]["category"] == "technical_indicators" for name in CATALOG_NAMES)
    assert all(formulas[name]["family"] for name in CATALOG_NAMES)
    assert all(formulas[name]["aliases"] for name in CATALOG_NAMES)
    assert all(formulas[name]["catalog_revision"] == 2 for name in CATALOG_NAMES)
    assert formulas["ta_macd_histogram_2x"]["status"] == "retired"
    assert formulas["ta_macd_histogram_2x"]["replacement"] == "ta_macd_histogram"
    macd = formulas["ta_macd_histogram"]
    assert [item["name"] for item in macd["inputs"]] == ["fast", "slow"]
    assert macd["constraints"] == [
        {"left": "fast", "operator": "lt", "right": "slow"}
    ]
    assert "mea" in formulas["ta_dea"]["aliases"]
    assert "Looking for MEA" in formulas["ta_dea"]["description"]

    saved = paths.formulas_path().read_bytes()
    second = client.get("/formulas")
    assert second.status_code == 200
    assert paths.formulas_path().read_bytes() == saved
    assert {item["name"] for item in second.json()} == formulas.keys()

    primitives = {item["name"]: item for item in client.get("/primitives").json()}
    assert primitives["ta_dea"]["origin"] == "catalog_formula"
    assert primitives["ta_dea"]["editable"] is False
    assert primitives["ta_dea"]["aliases"] == formulas["ta_dea"]["aliases"]


def test_catalog_v2_uses_canonical_fields_and_bounded_formula_parameters(
    client, synthetic_panel
):
    from alphalineage.core.extensions import expand_all
    from alphalineage.core.generate import RandomTreeGenerator
    from alphalineage.core.gp import GP, GPConfig, iter_variation_positions
    from alphalineage.core.tree import InvalidTree, Node, validate
    from alphalineage.core.types import DType

    formulas = {item["name"]: item for item in client.get("/formulas").json()}
    macd = formulas["ta_macd_histogram"]
    assert all(item["role"] == "parameter" for item in macd["inputs"])
    assert not any(item["type"] == "series" for item in macd["inputs"])
    assert {node["name"] for node in macd["body"]["children"]} == {
        formulas["ta_dif"]["runtime_name"],
        formulas["ta_dea"]["runtime_name"],
    }

    runtime = macd["runtime_name"]
    default = Node(runtime, (Node("window", value=12), Node("window", value=26)))
    validate(default)
    with pytest.raises(InvalidTree, match="requires fast < slow"):
        validate(Node(runtime, (Node("window", value=20), Node("window", value=20))))

    generator = RandomTreeGenerator(
        max_depth=8,
        max_nodes=40,
        root_type=DType.SERIES,
        allowed_operators={runtime},
    )
    assert generator.generate(grow=False) == default
    assert [path for path, _, _ in iter_variation_positions(default, DType.SERIES)] == [()]

    gp = GP(
        GPConfig(population_size=8, generations=1, max_depth=32, max_nodes=40),
        synthetic_panel,
        root_type=DType.SERIES,
        allowed_operators={runtime},
    )
    observed: set[tuple[int, int]] = set()
    for seed in range(40):
        gp.rng.seed(seed)
        mutated = gp._point_mutation(default)
        validate(mutated)
        observed.add(tuple(int(child.value) for child in mutated.children))
    assert observed - {(12, 26)}
    assert all(fast < slow for fast, slow in observed)
    assert (
        expand_all(default, max_depth=32, max_nodes=40).unique_computation_size()
        <= 40
    )

    standalone = Node("mul_scalar", (default, Node("const", value=1.0)))
    assert gp._mutate_formula_coefficient(standalone, (1,), standalone.children[1]) == standalone
    combination = Node("add", (standalone, Node("close")))
    weighted = gp._mutate_formula_coefficient(
        combination, (0, 1), standalone.children[1]
    )
    assert weighted is not None and weighted != combination


def test_every_active_catalog_formula_fits_default_node_budget(client):
    from alphalineage.core.extensions import expand_all
    from alphalineage.core.tree import Node

    formulas = client.get("/formulas").json()
    for formula in formulas:
        if formula["origin"] != "catalog_formula" or formula["status"] != "active":
            continue
        children = tuple(
            Node("window", value=int(item["default"]))
            if item["type"] == "window"
            else Node("const", value=float(item["default"]))
            for item in formula["inputs"]
        )
        expanded = expand_all(
            Node(formula["runtime_name"], children), max_depth=32, max_nodes=40
        )
        assert expanded.unique_computation_size() <= 40, formula["name"]


def test_deep_atomic_starters_initialize_and_mutate_with_default_gp_budget(
    client, synthetic_panel, tmp_path
):
    import json

    from alphalineage.core.gp import GP, GPConfig
    from alphalineage.core.tree import Node, validate
    from alphalineage.core.types import DType

    formulas = {item["name"]: item for item in client.get("/formulas").json()}
    selected = ["ta_rsi_wilder", "ta_kdj_j", "ta_adx", "ta_mfi"]
    seeds = []
    runtimes = set()
    for name in selected:
        formula = formulas[name]
        runtimes.add(formula["runtime_name"])
        children = tuple(
            Node("window", value=int(item["default"]))
            if item["type"] == "window"
            else Node("const", value=float(item["default"]))
            for item in formula["inputs"]
        )
        seeds.append(Node(formula["runtime_name"], children))

    gp = GP(
        GPConfig(population_size=8, generations=1),
        synthetic_panel,
        root_type=DType.SERIES,
        allowed_operators=runtimes,
    )
    assert all(gp._fits_complexity(tree) for tree in seeds)
    gp.initialize(seeds)
    checkpoint = tmp_path / "catalog-policy-checkpoint.json"
    gp.save_checkpoint(checkpoint)
    persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert runtimes <= persisted["formula_policies"].keys()
    assert persisted["evolution_version"] >= 2
    for tree in seeds:
        gp.rng.seed(17)
        mutated = gp._point_mutation(tree)
        validate(mutated)
        assert gp._fits_complexity(mutated)


def test_formula_allow_list_selects_one_active_starter(client):
    from alphalineage.api.app import _allowed_operators
    from alphalineage.core.categories import TECHNICAL_INDICATORS
    from alphalineage.core.gp import GPConfig

    formulas = {item["name"]: item for item in client.get("/formulas").json()}
    runtimes = {item["runtime_name"] for item in formulas.values()}
    config = GPConfig(
        population_size=8,
        generations=1,
        enabled_categories=[TECHNICAL_INDICATORS],
        enabled_formula_names=["ta_macd_histogram"],
    )
    allowed = _allowed_operators(config, runtimes)
    assert formulas["ta_macd_histogram"]["runtime_name"] in allowed
    assert formulas["ta_dif"]["runtime_name"] not in allowed
    assert formulas["ta_macd_histogram_2x"]["runtime_name"] not in allowed


def test_packaged_indicator_catalog_rejects_mutation_and_deletion(client):
    formula = next(item for item in client.get("/formulas").json() if item["name"] == "ta_sma")
    changed = {**formula, "description": "attempted replacement"}
    response = client.put("/formulas/ta_sma", json=changed)
    assert response.status_code == 403
    assert client.delete("/formulas/ta_sma").status_code == 403
    assert client.delete("/operators/ta_sma").status_code == 403


def test_catalog_upgrade_publishes_pinned_revision_and_preserves_user_formulas(
    client, monkeypatch
):
    import copy
    import importlib

    api_module = importlib.import_module("alphalineage.api.app")
    client.get("/formulas")
    user = {
        "name": "user_catalog_neighbor",
        "arg_types": [],
        "inputs": [],
        "out_type": "series",
        "body": {"name": "close"},
    }
    assert client.post("/formulas", json=user).status_code == 200

    upgraded_catalog = []
    for definition in api_module.INDICATOR_CATALOG:
        candidate = copy.deepcopy(definition)
        if candidate["name"] == "ta_ema":
            candidate["body"] = {
                "name": "ts_mean",
                "children": [{"name": "close"}, {"name": "$arg", "value": 0}],
            }
            candidate["catalog_revision"] = 3
        upgraded_catalog.append(candidate)
    monkeypatch.setattr(api_module, "INDICATOR_CATALOG", tuple(upgraded_catalog))

    assert client.get("/formulas").status_code == 200
    ema = client.get("/formulas/ta_ema").json()
    assert [item["runtime_name"] for item in ema["revisions"]] == ["ta_ema", "ta_ema__r2"]
    assert ema["body"]["name"] == "ts_mean"
    dema = client.get("/formulas/ta_dema").json()

    def names(tree):
        return {tree["name"]} | {
            name for child in tree.get("children", []) for name in names(child)
        }

    assert "ta_ema__r2" in names(dema["body"])
    user_detail = client.get("/formulas/user_catalog_neighbor").json()
    assert [item["runtime_name"] for item in user_detail["revisions"]] == [
        "user_catalog_neighbor"
    ]


def test_catalog_upgrade_retires_orphan_managed_family_without_rewriting_old_runtime(client):
    from alphalineage.data import paths

    legacy = {
        "schema_version": 2,
        "families": [
            {
                "name": "ta_rsi",
                "latest_revision": 1,
                "revisions": [
                    {
                        "name": "ta_rsi",
                        "display_name": "RSI",
                        "arg_types": [],
                        "inputs": [],
                        "out_type": "series",
                        "body": {"name": "close"},
                        "category": "technical_indicators",
                        "revision": 1,
                        "runtime_name": "ta_rsi",
                        "origin": "catalog_formula",
                        "editable": False,
                        "family": "rsi",
                        "aliases": ["rsi legacy"],
                        "catalog_revision": 1,
                    }
                ],
            }
        ],
    }
    paths.formulas_path().parent.mkdir(parents=True, exist_ok=True)
    paths.formulas_path().write_text(json.dumps(legacy), encoding="utf-8")

    formulas = client.get("/formulas").json()
    orphan = next(item for item in formulas if item["name"] == "ta_rsi")
    assert orphan["status"] == "retired"
    assert orphan["replacement"] == "ta_rsi_wilder"
    assert sum(
        item["origin"] == "catalog_formula" and item["status"] == "active"
        for item in formulas
    ) == 36
    detail = client.get("/formulas/ta_rsi").json()
    assert [item["runtime_name"] for item in detail["revisions"]] == [
        "ta_rsi",
        "ta_rsi__r2",
    ]
    assert detail["revisions"][0]["status"] == "active"


def test_packaged_indicators_expand_and_match_python_and_native(client, synthetic_panel):
    import numpy as np

    from alphalineage.core import cpp
    from alphalineage.core.evaluate import evaluate_python
    from alphalineage.core.extensions import expand_all
    from alphalineage.core.tree import Node

    formulas = client.get("/formulas").json()
    catalog = [item for item in formulas if item["origin"] == "catalog_formula"]
    assert {item["name"] for item in catalog} == CATALOG_NAMES

    for formula in catalog:
        children = []
        for item in formula["inputs"]:
            if item["type"] == "series":
                field = item["name"] if item["name"] in {"high", "low", "close"} else "close"
                children.append(Node(field))
            elif item["type"] == "window":
                children.append(Node("window", value=int(item["default"])))
            else:
                children.append(Node("const", value=float(item["default"])))
        compact = Node(formula["runtime_name"], tuple(children))
        expanded = expand_all(compact)
        expected = evaluate_python(expanded, synthetic_panel)
        actual = evaluate_python(compact, synthetic_panel)
        assert np.allclose(actual, expected, equal_nan=True), formula["name"]
        assert cpp.flatten(expanded) is not None, formula["name"]
        native = cpp.evaluate_cpp(compact, synthetic_panel)
        if native is not None:
            assert np.allclose(
                native, expected, equal_nan=True, rtol=1e-7, atol=1e-9
            ), formula["name"]


def test_wilder_rsi_and_kdj_define_flat_and_zero_loss_edges(client):
    import numpy as np
    import pandas as pd

    from alphalineage.core.evaluate import evaluate_python
    from alphalineage.core.panel import Panel
    from alphalineage.core.tree import Node

    client.get("/formulas")  # seed/register managed formulas
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    columns = ["FLAT", "UP", "DOWN"]
    close = pd.DataFrame(
        {
            "FLAT": np.full(8, 10.0),
            "UP": np.arange(10.0, 18.0),
            "DOWN": np.arange(18.0, 10.0, -1.0),
        },
        index=dates,
    )
    panel = Panel.from_prices(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=pd.DataFrame(1.0, index=dates, columns=columns),
    )
    rsi = evaluate_python(
        Node("ta_rsi_wilder", (Node("window", value=3),)),
        panel,
    )
    assert np.isnan(rsi.iloc[:3].to_numpy()).all()
    assert np.allclose(rsi["FLAT"].iloc[3:], 50.0)
    assert np.allclose(rsi["UP"].iloc[3:], 100.0)
    assert np.allclose(rsi["DOWN"].iloc[3:], 0.0)

    rsv = evaluate_python(
        Node(
            "ta_kdj_rsv",
            (Node("window", value=3),),
        ),
        panel,
    )
    assert np.isnan(rsv.iloc[:2].to_numpy()).all()
    assert np.allclose(rsv["FLAT"].iloc[2:], 50.0)


def test_indicator_families_match_independent_fixed_references(client):
    import numpy as np
    import pandas as pd

    from alphalineage.core.evaluate import evaluate_python
    from alphalineage.core.panel import Panel
    from alphalineage.core.tree import Node

    client.get("/formulas")
    dates = pd.date_range("2024-02-01", periods=12, freq="B")
    close = pd.Series(
        [10.0, 11.0, 10.0, 12.0, 12.0, 13.0, 11.0, 14.0, 15.0, 14.0, 16.0, 15.0],
        index=dates,
        name="A",
    )
    high = close + pd.Series(
        [1.0, 1.5, 0.8, 1.2, 1.0, 1.4, 0.9, 1.3, 1.1, 0.7, 1.6, 1.0],
        index=dates,
    )
    low = close - pd.Series(
        [0.7, 0.8, 1.1, 0.6, 1.2, 0.7, 1.0, 0.9, 0.6, 1.3, 0.8, 1.1],
        index=dates,
    )
    frame = lambda values: values.to_frame("A")  # noqa: E731 - compact fixture adapter
    panel = Panel.from_prices(
        open=frame(close.shift(1).fillna(close.iloc[0])),
        high=frame(high),
        low=frame(low),
        close=frame(close),
        volume=pd.DataFrame(1.0, index=dates, columns=["A"]),
    )

    def result(name: str, *children: Node) -> pd.Series:
        return evaluate_python(Node(name, tuple(children)), panel)["A"]

    def window(value: int) -> Node:
        return Node("window", value=value)

    def scalar(value: float) -> Node:
        return Node("const", value=value)

    def wilder(source: pd.Series, lookback: int) -> pd.Series:
        output = pd.Series(np.nan, index=source.index, dtype=float)
        seed: list[float] = []
        state: float | None = None
        for index, value in source.items():
            if not np.isfinite(value):
                seed.clear()
                state = None
                continue
            if state is None:
                seed.append(float(value))
                if len(seed) < lookback:
                    continue
                state = sum(seed) / lookback
                seed.clear()
            else:
                state = (state * (lookback - 1) + float(value)) / lookback
            output.loc[index] = state
        return output

    def recursive(source: pd.Series, lookback: int, initial: float = 50.0) -> pd.Series:
        output = pd.Series(np.nan, index=source.index, dtype=float)
        state = initial
        for index, value in source.items():
            if not np.isfinite(value):
                continue
            state = (state * (lookback - 1) + float(value)) / lookback
            output.loc[index] = state
        return output

    expected_sma = close.rolling(3, min_periods=3).mean()
    expected_ema = close.ewm(span=3, adjust=False, min_periods=3).mean()
    expected_rma = wilder(close, 3)
    assert np.allclose(result("ta_sma", window(3)), expected_sma, equal_nan=True)
    assert np.allclose(result("ta_ema", window(3)), expected_ema, equal_nan=True)
    assert np.allclose(result("ta_rma", window(3)), expected_rma, equal_nan=True)

    fast = close.ewm(span=2, adjust=False, min_periods=2).mean()
    slow = close.ewm(span=3, adjust=False, min_periods=3).mean()
    expected_dif = fast - slow
    expected_dea = expected_dif.ewm(span=9, adjust=False, min_periods=9).mean()
    expected_macd = expected_dif - expected_dea
    macd_args = (window(2), window(3))
    assert np.allclose(result("ta_dif", *macd_args), expected_dif, equal_nan=True)
    assert np.allclose(result("ta_dea", *macd_args), expected_dea, equal_nan=True)
    assert np.allclose(
        result("ta_macd_histogram", *macd_args), expected_macd, equal_nan=True
    )
    assert np.allclose(
        result("ta_macd_histogram_2x", *macd_args), expected_macd * 2.0, equal_nan=True
    )

    middle = close.rolling(3, min_periods=3).mean()
    deviation = close.rolling(3, min_periods=3).std(ddof=0)
    upper, lower = middle + 2.0 * deviation, middle - 2.0 * deviation
    percent_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100.0
    boll_args = (window(3),)
    for name, expected in (
        ("ta_boll_middle", middle),
        ("ta_boll_upper", upper),
        ("ta_boll_lower", lower),
        ("ta_boll_percent_b", percent_b),
        ("ta_boll_bandwidth", bandwidth),
    ):
        assert np.allclose(result(name, *boll_args), expected, equal_nan=True), name

    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1, skipna=True)
    expected_atr = wilder(true_range, 3)
    assert np.allclose(result("ta_true_range"), true_range, equal_nan=True)
    assert np.allclose(result("ta_atr", window(3)), expected_atr, equal_nan=True)
    expected_roc = (close / close.shift(2) - 1.0) * 100.0
    assert np.allclose(
        result("ta_roc", window(2)), expected_roc, equal_nan=True
    )

    lowest = low.rolling(3, min_periods=3).min()
    spread = high.rolling(3, min_periods=3).max() - lowest
    expected_rsv = ((close - lowest) / spread * 100.0).mask(spread == 0.0, 50.0)
    expected_k = recursive(expected_rsv, 3)
    expected_d = recursive(expected_k, 3)
    expected_j = 3.0 * expected_k - 2.0 * expected_d
    assert np.allclose(result("ta_kdj_rsv", window(3)), expected_rsv, equal_nan=True)
    assert np.allclose(result("ta_kdj_k", window(3)), expected_k, equal_nan=True)
    assert np.allclose(result("ta_kdj_d", window(3)), expected_d, equal_nan=True)
    assert np.allclose(result("ta_kdj_j", window(3)), expected_j, equal_nan=True)


def test_catalog_v2_additions_match_independent_references(client):
    import numpy as np
    import pandas as pd

    from alphalineage.core.evaluate import evaluate_python
    from alphalineage.core.panel import Panel
    from alphalineage.core.tree import Node

    client.get("/formulas")
    dates = pd.date_range("2024-04-01", periods=30, freq="B")
    close = pd.Series(
        [
            10, 11, 10, 12, 13, 12, 14, 15, 14, 16,
            15, 17, 18, 17, 19, 20, 18, 21, 22, 21,
            23, 22, 24, 25, 24, 26, 27, 25, 28, 29,
        ],
        index=dates,
        dtype=float,
    )
    high = close + pd.Series(np.resize([1.0, 1.4, 0.8, 1.2], 30), index=dates)
    low = close - pd.Series(np.resize([0.7, 1.1, 0.9], 30), index=dates)
    volume = pd.Series(np.arange(100.0, 130.0), index=dates)
    frame = lambda values: values.to_frame("A")  # noqa: E731
    panel = Panel.from_prices(
        open=frame(close.shift(1).fillna(close.iloc[0])),
        high=frame(high),
        low=frame(low),
        close=frame(close),
        volume=frame(volume),
    )

    def result(name: str, *values: int) -> pd.Series:
        return evaluate_python(
            Node(name, tuple(Node("window", value=value) for value in values)), panel
        )["A"]

    def wilder(source: pd.Series, lookback: int) -> pd.Series:
        output = pd.Series(np.nan, index=source.index, dtype=float)
        seed: list[float] = []
        state: float | None = None
        for index, value in source.items():
            if not np.isfinite(value):
                seed.clear()
                state = None
                continue
            if state is None:
                seed.append(float(value))
                if len(seed) < lookback:
                    continue
                state = sum(seed) / lookback
                seed.clear()
            else:
                state = (state * (lookback - 1) + float(value)) / lookback
            output.loc[index] = state
        return output

    weights = np.arange(1.0, 4.0)
    wma = close.rolling(3, min_periods=3).apply(
        lambda values: float(np.dot(values, weights) / weights.sum()), raw=True
    )
    ema1 = close.ewm(span=3, adjust=False, min_periods=3).mean()
    ema2 = ema1.ewm(span=3, adjust=False, min_periods=3).mean()
    ema3 = ema2.ewm(span=3, adjust=False, min_periods=3).mean()
    assert np.allclose(result("ta_wma", 3), wma, equal_nan=True)
    assert np.allclose(result("ta_dema", 3), 2 * ema1 - ema2, equal_nan=True)
    assert np.allclose(result("ta_tema", 3), 3 * ema1 - 3 * ema2 + ema3, equal_nan=True)

    fast = close.ewm(span=2, adjust=False, min_periods=2).mean()
    slow = close.ewm(span=5, adjust=False, min_periods=5).mean()
    assert np.allclose(result("ta_ppo", 2, 5), (fast - slow) / slow * 100, equal_nan=True)

    highest = high.rolling(3, min_periods=3).max()
    lowest = low.rolling(3, min_periods=3).min()
    assert np.allclose(
        result("ta_williams_r", 3), (highest - close) / (highest - lowest) * -100,
        equal_nan=True,
    )
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1, skipna=True)
    atr = wilder(true_range, 3)
    assert np.allclose(result("ta_natr", 3), atr / close * 100, equal_nan=True)
    assert np.allclose(result("ta_donchian_upper", 3), highest, equal_nan=True)
    assert np.allclose(result("ta_donchian_lower", 3), lowest, equal_nan=True)
    assert np.allclose(
        result("ta_donchian_middle", 3), (highest + lowest) / 2, equal_nan=True
    )
    assert np.allclose(
        result("ta_donchian_position", 3), (close - lowest) / (highest - lowest) * 100,
        equal_nan=True,
    )

    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    plus_di = wilder(plus_dm, 3) / atr * 100
    minus_di = wilder(minus_dm, 3) / atr * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = wilder(dx, 3)
    assert np.allclose(result("ta_plus_di", 3), plus_di, equal_nan=True)
    assert np.allclose(result("ta_minus_di", 3), minus_di, equal_nan=True)
    assert np.allclose(result("ta_dx", 3), dx, equal_nan=True)
    assert np.allclose(result("ta_adx", 3), adx, equal_nan=True)

    signed_volume = np.sign(close.diff()) * volume
    assert np.allclose(result("ta_obv"), signed_volume.cumsum(), equal_nan=True)
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    change = typical.diff()
    positive = raw_flow.where(change > 0, 0.0).rolling(3, min_periods=3).sum()
    negative = raw_flow.where(change < 0, 0.0).rolling(3, min_periods=3).sum()
    mfi = 100 - 100 / (1 + positive / negative)
    mfi = mfi.mask((negative == 0) & (positive > 0), 100.0)
    mfi = mfi.mask((negative == 0) & (positive == 0), 50.0)
    assert np.allclose(result("ta_mfi", 3), mfi, equal_nan=True)
    multiplier = ((close - low) - (high - close)) / (high - low)
    cmf = (multiplier * volume).rolling(3, min_periods=3).sum() / volume.rolling(
        3, min_periods=3
    ).sum()
    assert np.allclose(result("ta_cmf", 3), cmf, equal_nan=True)


def test_indicator_zero_denominators_are_explicitly_nan(client):
    import numpy as np
    import pandas as pd

    from alphalineage.core.evaluate import evaluate_python
    from alphalineage.core.panel import Panel
    from alphalineage.core.tree import Node

    client.get("/formulas")
    dates = pd.date_range("2024-03-01", periods=6, freq="B")
    close = pd.DataFrame(
        {
            "ZERO": np.zeros(6),
            "FLAT": np.full(6, 10.0),
            "CROSS": [0.0, 0.0, 1.0, 0.0, 2.0, 2.0],
        },
        index=dates,
    )
    panel = Panel.from_prices(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=pd.DataFrame(1.0, index=dates, columns=close.columns),
    )
    window = Node("window", value=3)
    boll_args = (window,)
    percent_b = evaluate_python(Node("ta_boll_percent_b", boll_args), panel)
    bandwidth = evaluate_python(Node("ta_boll_bandwidth", boll_args), panel)
    assert percent_b[["ZERO", "FLAT"]].iloc[2:].isna().all().all()
    assert bandwidth["ZERO"].iloc[2:].isna().all()  # zero width / zero middle
    assert np.allclose(bandwidth["FLAT"].iloc[2:], 0.0)

    roc = evaluate_python(
        Node("ta_roc", (Node("window", value=1),)), panel
    )
    assert roc["ZERO"].iloc[1:].isna().all()
    assert roc["CROSS"].iloc[1:].isna().to_list() == [True, True, False, True, False]
    assert roc["CROSS"].iloc[3] == -100.0

    cmf = evaluate_python(Node("ta_cmf", (window,)), panel)
    assert np.allclose(cmf.iloc[2:].to_numpy(), 0.0)


def test_legacy_formula_store_migrates_named_inputs_on_write(client):
    from alphalineage.data import paths

    legacy = {
        "name": "legacy_ma",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    _write_formulas([legacy])
    loaded = client.get("/formulas").json()[0]
    assert [item["name"] for item in loaded["inputs"]] == ["input_1", "input_2"]
    assert loaded["revision"] == 1
    assert loaded["runtime_name"] == "legacy_ma"

    loaded["description"] = "Readable moving average."
    assert client.put("/formulas/legacy_ma", json=loaded).status_code == 200
    payload = json.loads(paths.formulas_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["families"][0]["revisions"][0]["inputs"][0]["name"] == "input_1"


def test_formula_impact_and_transitive_upgrade_preserve_old_revisions(client):
    ma = {
        "name": "ma_shared",
        "description": "Moving average.",
        "arg_types": ["series", "window"],
        "inputs": [
            {"name": "price", "type": "series", "description": "Input series."},
            {"name": "lookback", "type": "window", "description": "Period count."},
        ],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    dif = {
        "name": "dif_shared",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ma_shared",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    signal = {
        "name": "signal_shared",
        "arg_types": ["series", "window"],
        "out_type": "signal",
        "body": {
            "name": "rank",
            "children": [
                {
                    "name": "dif_shared",
                    "children": [
                        {"name": "$arg", "value": 0},
                        {"name": "$arg", "value": 1},
                    ],
                }
            ],
        },
    }
    for spec in (ma, dif, signal):
        assert client.post("/formulas", json=spec).status_code == 200

    changed = {
        **ma,
        "body": {
            "name": "delta",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    impact = client.post("/formulas/ma_shared/impact", json=changed).json()
    assert impact["direct_formulas"] == ["dif_shared"]
    assert impact["transitive_formulas"] == ["dif_shared", "signal_shared"]
    assert client.put("/formulas/ma_shared", json=changed).status_code == 400

    upgraded = client.put("/formulas/ma_shared?strategy=upgrade_references", json=changed)
    assert upgraded.status_code == 200, upgraded.text
    assert upgraded.json()["revision"] == 2
    assert upgraded.json()["upgraded"] == ["dif_shared", "signal_shared"]

    ma_detail = client.get("/formulas/ma_shared").json()
    assert [item["runtime_name"] for item in ma_detail["revisions"]] == [
        "ma_shared",
        "ma_shared__r2",
    ]
    dif_detail = client.get("/formulas/dif_shared").json()
    assert dif_detail["body"]["name"] == "ma_shared__r2"
    signal_detail = client.get("/formulas/signal_shared").json()
    assert signal_detail["body"]["children"][0]["name"] == "dif_shared__r2"
    assert "ma_shared" in REGISTRY and "ma_shared__r2" in REGISTRY

    # A process restart must restore immutable runtime revision names, not overwrite the
    # base family name with each historical body.
    extensions.clear_user_operators()
    reloaded = client.get("/formulas")
    assert reloaded.status_code == 200
    assert "ma_shared" in REGISTRY and "ma_shared__r2" in REGISTRY
    assert "dif_shared__r2" in REGISTRY and "signal_shared__r2" in REGISTRY


def test_saved_macd_can_be_nested_saved_reused_and_reloaded_with_evaluator_parity(
    client, synthetic_panel
):
    """Acceptance path: saved formula -> MA(10) -> saved formula -> another formula."""
    import pandas as pd

    from alphalineage.core import cpp
    from alphalineage.core.evaluate import evaluate_python
    from alphalineage.core.tree import Node

    macd = {
        "name": "macd_nested_test",
        "arg_types": [],
        "inputs": [],
        "out_type": "series",
        "body": {
            "name": "sub",
            "children": [
                {
                    "name": "ts_ema",
                    "children": [
                        {"name": "close"},
                        {"name": "window", "value": 12},
                    ],
                },
                {
                    "name": "ts_ema",
                    "children": [
                        {"name": "close"},
                        {"name": "window", "value": 26},
                    ],
                },
            ],
        },
    }
    macd_ma = {
        "name": "macd_ma10_nested_test",
        "arg_types": [],
        "inputs": [],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [
                {"name": "macd_nested_test"},
                {"name": "window", "value": 10},
            ],
        },
    }
    reused = {
        "name": "macd_ma10_rank_nested_test",
        "arg_types": [],
        "inputs": [],
        "out_type": "signal",
        "body": {
            "name": "rank",
            "children": [{"name": "macd_ma10_nested_test"}],
        },
    }
    for spec in (macd, macd_ma, reused):
        response = client.post("/formulas", json=spec)
        assert response.status_code == 200, response.text

    extensions.clear_user_operators()
    assert client.get("/formulas").status_code == 200
    assert {macd["name"], macd_ma["name"], reused["name"]} <= set(REGISTRY)

    compact = Node(reused["name"])
    expanded = extensions.expand_all(compact)
    compact_python = evaluate_python(compact, synthetic_panel)
    expanded_python = evaluate_python(expanded, synthetic_panel)
    pd.testing.assert_frame_equal(compact_python, expanded_python)

    native = cpp.evaluate_cpp(compact, synthetic_panel)
    if native is not None:
        pd.testing.assert_frame_equal(native, expanded_python, rtol=1e-7, atol=1e-9)


# --- Stage 3: boolean / condition ops --------------------------------------------
_CONDITION_OPS = {"gt", "lt", "ge", "le", "and_", "or_", "not_", "where"}


def test_where_tree_evaluates_via_public_evaluate(synthetic_panel):
    # `where(gt(close, open), close, open)` == elementwise max(close, open) wherever both exist.
    import numpy as np

    from alphalineage.core.evaluate import evaluate
    from alphalineage.core.tree import Node, validate

    tree = Node(
        "where",
        (
            Node("gt", (Node("close"), Node("open"))),
            Node("close"),
            Node("open"),
        ),
    )
    validate(tree)  # type-checks: gt -> bool, where(bool, series, series) -> series
    result = evaluate(tree, synthetic_panel)
    close, open_ = synthetic_panel["close"], synthetic_panel["open"]
    expected = close.where(close > open_, open_)
    assert np.allclose(result.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_default_generator_excludes_condition_ops():
    import random

    from alphalineage.core.generate import GenerationError, RandomTreeGenerator
    from alphalineage.core.types import DType

    gen = RandomTreeGenerator(random.Random(0), max_depth=5, max_nodes=30)  # allowed=None default
    # bool is unreachable by default (no condition producers, no bool terminal)
    with pytest.raises(GenerationError):
        gen.grow_subtree(DType.BOOL, max_depth=4, max_nodes=12)
    # and no generated tree ever contains a condition op
    for _ in range(80):
        tree = gen.generate()
        assert not (_CONDITION_OPS & {n.name for n in tree.iter_nodes()})


def test_enabled_generator_can_build_conditions():
    import random

    from alphalineage.core.generate import RandomTreeGenerator
    from alphalineage.core.primitives import OPERATORS
    from alphalineage.core.tree import validate
    from alphalineage.core.types import DType

    allowed = set(OPERATORS)  # everything, including the condition category
    gen = RandomTreeGenerator(
        random.Random(0), max_depth=5, max_nodes=30, allowed_operators=allowed
    )

    bool_tree = gen.grow_subtree(DType.BOOL, max_depth=4, max_nodes=12)
    validate(bool_tree)
    assert bool_tree.out_type is DType.BOOL

    # over many full trees, `where` (and a comparison) should appear at least once
    names: set[str] = set()
    for _ in range(200):
        names |= {n.name for n in gen.generate().iter_nodes()}
    assert "where" in names
