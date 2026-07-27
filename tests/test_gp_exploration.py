"""Focused coverage for normalized complexity and managed-formula exploration."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from alphalineage.core.extensions import ARG, register_operator, unregister_operator
from alphalineage.core.fitness import forward_returns, score_tree
from alphalineage.core.gp import (
    GP,
    PARAMETER_BEAM_WIDTH,
    GPConfig,
    Individual,
    resolve_exploration_settings,
)
from alphalineage.core.tree import Node, to_json, validate
from alphalineage.core.types import DType


def _managed_policy(fast: int, slow: int) -> dict[str, object]:
    def parameter(name: str, default: int, minimum: int, maximum: int) -> dict[str, object]:
        return {
            "name": name,
            "type": "window",
            "default": default,
            "role": "parameter",
            "tuning": {
                "enabled": True,
                "min": minimum,
                "max": maximum,
                "step": 1,
                "radius": 2,
            },
        }

    return {
        "inputs": [
            parameter("fast", fast, 2, 20),
            parameter("slow", slow, 3, 40),
        ],
        "constraints": [{"left": "fast", "operator": "lt", "right": "slow"}],
        "status": "active",
        "catalog_revision": 1,
    }


@contextmanager
def _managed_macd_like(name: str, fast: int = 5, slow: int = 12):
    body = Node(
        "sub",
        (
            Node("ts_mean", (Node("close"), Node(ARG, value=0))),
            Node("ts_mean", (Node("close"), Node(ARG, value=1))),
        ),
    )
    register_operator(
        name,
        [DType.WINDOW, DType.WINDOW],
        DType.SERIES,
        body,
        policy=_managed_policy(fast, slow),
    )
    try:
        yield
    finally:
        unregister_operator(name)


def test_gp_config_preserves_legacy_parsimony_and_resolves_normalized_default():
    legacy = GPConfig(parsimony=0.017)
    assert legacy.complexity_penalty_mode == "per_node"
    assert legacy.complexity_penalty_value is None
    assert legacy.resolved_complexity_penalty_value == pytest.approx(0.017)

    normalized = GPConfig(complexity_penalty_mode="normalized_budget")
    assert normalized.resolved_complexity_penalty_value == pytest.approx(0.005)

    restored = GPConfig.from_dict(legacy.to_dict())
    assert restored == legacy
    with pytest.raises(ValueError, match="complexity_penalty_mode"):
        GPConfig(complexity_penalty_mode="unknown")
    with pytest.raises(ValueError, match="parameter_neighbor_fraction"):
        GPConfig(parameter_neighbor_fraction=1.01)
    with pytest.raises(ValueError, match="validation_folds"):
        GPConfig(validation_folds=1)


def test_normalized_complexity_penalty_is_bounded_by_node_budget(signal_panel):
    panel, _ = signal_panel
    tree = Node(
        "add",
        (
            Node("volume"),
            Node("mul_scalar", (Node("volume"), Node("const", value=0.0))),
        ),
    )
    fitness, metrics = score_tree(
        tree,
        panel,
        forward_returns(panel),
        complexity_penalty_mode="normalized_budget",
        complexity_penalty_value=0.005,
        max_nodes=40,
    )
    expected = 0.005 * tree.size() / 40
    assert metrics["complexity_penalty"] == pytest.approx(expected)
    assert metrics["expanded_complexity"] == tree.size()
    assert metrics["expanded_unique_nodes"] == tree.unique_computation_size()
    assert fitness == pytest.approx(metrics["raw_objective"] - expected)


def test_exploration_profiles_resolve_legacy_and_public_quotas():
    assert GPConfig().resolved_exploration_profile == "classic"
    assert resolve_exploration_settings(None).profile == "classic"
    balanced = resolve_exploration_settings("balanced")
    assert balanced.stepping_stone_fraction == pytest.approx(0.10)
    assert balanced.protected_parent_fraction == pytest.approx(0.20)
    assert balanced.two_edit_fraction == pytest.approx(0.10)
    aggressive = resolve_exploration_settings("aggressive")
    assert aggressive.stepping_stone_fraction == pytest.approx(0.20)
    assert aggressive.protected_parent_fraction == pytest.approx(0.40)
    assert aggressive.two_edit_fraction == pytest.approx(0.20)
    assert aggressive.parameter_step_multipliers == (1, 2, 4)
    boosted = aggressive.with_stagnation_boost(4)
    assert boosted.stepping_stone_fraction == pytest.approx(0.20)
    assert boosted.protected_parent_fraction == pytest.approx(0.50)
    assert boosted.two_edit_fraction == pytest.approx(0.30)
    with pytest.raises(ValueError, match="exploration_profile"):
        GPConfig(exploration_profile="legacy")


def test_pre_cache_simplification_deduplicates_semantic_identity(signal_panel):
    panel, _ = signal_panel
    gp = GP(GPConfig(population_size=4, generations=1), panel)
    volume = Node("volume")
    identity = Node(
        "mul_scalar",
        (volume, Node("const", value=1.0)),
    )
    individuals = gp._individuals(
        [identity, volume],
        phase="initializing",
        birth_generation=0,
    )
    assert [item.tree for item in individuals] == [volume, volume]
    assert len(gp._cache) == 1
    assert gp._pre_cache_simplifications >= 1


def test_stepping_stone_niche_cap_and_ttl(signal_panel):
    panel, _ = signal_panel
    gp = GP(
        GPConfig(
            population_size=6,
            generations=1,
            exploration_profile="aggressive",
        ),
        panel,
    )
    fields = ("close", "volume", "open", "high", "low", "returns")
    gp.population = [
        Individual(
            Node(name),
            float(10 - index),
            {},
            birth_generation=(0 if index < 3 else 3),
        )
        for index, name in enumerate(fields)
    ]
    gp.generation = 3
    settings = resolve_exploration_settings("aggressive")
    assert gp._exploration_niche(gp.population[0].tree) == (
        (),
        "data",
        "01-05",
    )
    selected = gp._stepping_stone_indices(
        list(range(6)),
        excluded=set(),
        target=6,
        settings=settings,
    )
    assert selected == [0, 3]
    gp.generation = 8
    assert (
        gp._stepping_stone_indices(
            list(range(6)),
            excluded=set(),
            target=6,
            settings=settings,
        )
        == []
    )


def test_native_normalized_complexity_matches_python(signal_panel):
    from alphalineage.core import cpp

    if not cpp.supports_native_scoring("spearman"):
        pytest.skip("native scoring extension unavailable")
    panel, _ = signal_panel
    tree = Node(
        "add",
        (
            Node("volume"),
            Node("mul_scalar", (Node("volume"), Node("const", value=0.0))),
        ),
    )
    target = forward_returns(panel)
    kwargs = {
        "complexity_penalty_mode": "normalized_budget",
        "complexity_penalty_value": 0.005,
        "max_nodes": 40,
    }
    expected = score_tree(tree, panel, target, **kwargs)
    actual = cpp.score_many([tree], panel, target, workers=2, **kwargs)[0]
    assert actual is not None
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert actual[1] == pytest.approx(expected[1], abs=1e-12)


def test_enabled_managed_default_and_parameter_neighbors_preserve_constraints(
    signal_panel,
):
    panel, _ = signal_panel
    name = "managed_neighbor_test"
    with _managed_macd_like(name):
        config = GPConfig(
            population_size=12,
            generations=2,
            max_depth=6,
            max_nodes=24,
            seed=71,
            enabled_formula_names=[name],
            parameter_neighbor_fraction=0.5,
            exploration_profile="aggressive",
        )
        gp = GP(config, panel, allowed_operators={name})
        gp.run()

        default = Node(
            name,
            (Node("window", value=5), Node("window", value=12)),
        )
        assert to_json(default) in gp._cache
        diagnostics = gp.formula_exploration()[name]
        assert diagnostics["default_evaluated"] == 1
        assert diagnostics["calls_searched"] >= 2
        assert diagnostics["distinct_parameter_tuples"] >= 2
        assert gp.history[-1]["parameter_neighbor_count"] > 0
        assert name in gp.history[-1]["exploration"][
            "parameter_distance_from_defaults"
        ]
        default_moves = gp._policy_move_candidates(
            default,
            (),
            multipliers=(1, 2, 4),
            include_diagonal=True,
        )
        tuples = {
            (candidate.children[0].value, candidate.children[1].value)
            for candidate, _kind in default_moves
        }
        assert {(6, 12), (4, 12), (7, 12), (3, 12), (9, 12)} <= tuples
        assert (6, 13) in tuples
        assert any(kind == "diagonal" for _candidate, kind in default_moves)
        assert (
            len(gp._parameter_beam_sources(width=PARAMETER_BEAM_WIDTH))
            <= PARAMETER_BEAM_WIDTH
        )
        gp.population = [
            Individual(
                Node(
                    name,
                    (
                        Node("window", value=fast),
                        Node("window", value=slow),
                    ),
                ),
                fitness,
            )
            for fast, slow, fitness in (
                (5, 12, 0.50),
                (6, 12, 0.40),
                (4, 12, 0.30),
                (7, 13, 0.20),
                (3, 10, 0.10),
            )
        ]
        beam = gp._parameter_beam_sources(width=PARAMETER_BEAM_WIDTH)
        assert len(beam) == 4
        assert [
            gp._parameter_tuple(gp._node_at(tree, path))
            for _formula, _index, tree, path in beam
        ] == [(5, 12), (6, 12), (4, 12), (7, 13)]
        assert len(gp._parameter_frontier_sources()) == 5
        for individual in gp.searched_individuals():
            for call in individual.tree.iter_nodes():
                if call.name != name:
                    continue
                validate(call)
                assert call.children[0].value < call.children[1].value


def test_catalog_overflow_defaults_are_injected_and_checkpointed(signal_panel, tmp_path):
    panel, _ = signal_panel
    names = [f"managed_coverage_{index}" for index in range(5)]
    contexts = [
        _managed_macd_like(name, fast=3 + index, slow=10 + index)
        for index, name in enumerate(names)
    ]
    with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4]:
        config = GPConfig(
            population_size=4,
            generations=1,
            elitism=1,
            max_depth=6,
            max_nodes=24,
            seed=82,
            enabled_formula_names=names,
            parameter_neighbor_fraction=0.25,
        )
        checkpoint = Path(tmp_path) / "managed-coverage.json"
        gp = GP(config, panel, allowed_operators=set(names), workers=1)
        gp.run(checkpoint_path=checkpoint)
        first_default = Node(
            names[0],
            (Node("window", value=3), Node("window", value=10)),
        )
        gp.record_formula_validation_scores([(first_default, 0.123)])
        gp.save_checkpoint(checkpoint)

        diagnostics = gp.formula_exploration()
        assert all(diagnostics[name]["default_evaluated"] == 1 for name in names)
        assert diagnostics[names[0]]["best_validation_score"] == pytest.approx(0.123)
        assert gp.history[-1]["formula_default_injection_count"] >= 1
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert payload["formula_exploration"] == diagnostics
        assert payload["formula_exploration_state"]["pending_defaults"] == []
        assert payload["resolved_exploration"]["profile"] == "classic"
        assert "parameter_frontier_cursor" in payload["exploration_state"]

        restored = GP.from_checkpoint(
            checkpoint,
            panel,
            allowed_operators=set(names),
            workers=4,
        )
        assert restored.formula_exploration() == diagnostics


def test_parameter_trajectory_is_worker_and_resume_deterministic(signal_panel, tmp_path):
    panel, _ = signal_panel
    name = "managed_determinism_test"
    with _managed_macd_like(name):
        config = GPConfig(
            population_size=12,
            generations=2,
            max_depth=6,
            max_nodes=24,
            seed=93,
            enabled_formula_names=[name],
            parameter_neighbor_fraction=0.5,
            exploration_profile="aggressive",
        )
        serial = GP(config, panel, allowed_operators={name}, workers=1)
        parallel = GP(config, panel, allowed_operators={name}, workers=4)
        serial.run()
        parallel.run()

        checkpoint = Path(tmp_path) / "parameter-resume.json"
        partial = GP(config, panel, allowed_operators={name}, workers=2)
        partial.run(generations=1, checkpoint_path=checkpoint)
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert checkpoint_payload["resolved_exploration"]["profile"] == "aggressive"
        assert all(
            "birth_generation" in individual
            and "stepping_stone_ancestors" in individual
            for individual in checkpoint_payload["population"]
        )
        resumed = GP.from_checkpoint(
            checkpoint,
            panel,
            allowed_operators={name},
            workers=4,
        )
        resumed.run()

        expected = [to_json(item.tree) for item in serial.population]
        assert [to_json(item.tree) for item in parallel.population] == expected
        assert [to_json(item.tree) for item in resumed.population] == expected
        assert serial.history == parallel.history == resumed.history
        assert serial.formula_exploration() == parallel.formula_exploration()
        assert serial.formula_exploration() == resumed.formula_exploration()


def test_aggressive_profile_records_protected_exploration_lanes(signal_panel):
    panel, _ = signal_panel
    config = GPConfig(
        population_size=24,
        generations=1,
        max_depth=6,
        max_nodes=40,
        seed=109,
        exploration_profile="aggressive",
    )
    gp = GP(config, panel)
    gp.run()
    diagnostics = gp.history[-1]["exploration"]
    assert diagnostics["profile"] == "aggressive"
    assert diagnostics["stepping_stone_survivors"] >= 1
    assert diagnostics["protected_parent_offspring"] >= 1
    assert diagnostics["two_edit_attempts"] >= 1
    assert diagnostics["composition_attempts"] >= 1
    assert diagnostics["insertion_attempts"] >= 1
    assert diagnostics["operation_attempts"]
    assert diagnostics["niche_occupancy"]
    assert diagnostics["parent_contribution"]
    assert 0.0 <= diagnostics["no_op_crossover_rate"] <= 1.0
    assert "parameter_distance_from_defaults" in diagnostics
    assert "champion_stepping_stone_ancestors" in diagnostics


def test_composition_crossover_uses_only_approved_intact_operators(signal_panel):
    panel, _ = signal_panel
    left, right = Node("close"), Node("volume")
    blocked = GP(
        GPConfig(population_size=4, generations=1, exploration_profile="aggressive"),
        panel,
        allowed_operators={"div"},
    )
    assert blocked._composition_crossover(left, right) == left
    allowed = GP(
        GPConfig(population_size=4, generations=1, exploration_profile="aggressive"),
        panel,
        allowed_operators={"add"},
    )
    assert allowed._composition_crossover(left, right) == Node(
        "add", (left, right)
    )
