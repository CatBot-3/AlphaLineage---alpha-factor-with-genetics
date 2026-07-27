"""Phase 2 acceptance + supporting tests for the GP loop (tests/test_gp.py)."""

from __future__ import annotations

import pytest

from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import mean_ic
from alphalineage.core.gp import (
    EVOLUTION_VERSION,
    GP,
    MAX_GENERATIONS,
    MAX_HORIZON,
    MAX_MIN_NAMES,
    MAX_POPULATION_SIZE,
    MAX_TIME_BUDGET_S,
    MAX_TREE_DEPTH,
    MAX_TREE_NODES,
    SCORER_VERSION,
    GPConfig,
    TrainingCancelled,
    iter_positions,
    replace_at,
)
from alphalineage.core.tree import Node, is_valid, to_json
from alphalineage.core.types import DType


def test_synthetic_signal_recovery(signal_panel):
    """Load-bearing: GP recovers a factor highly correlated with the injected signal."""
    panel, injected = signal_panel
    config = GPConfig(
        population_size=80, generations=12, max_depth=5, max_nodes=30, seed=1, parsimony=1e-3
    )
    best = GP(config, panel).run()

    factor = evaluate(best.tree, panel)
    recovered_ic = mean_ic(factor, injected, "spearman", absolute=True)
    assert recovered_ic > 0.7, recovered_ic


def test_train_ic_improves(signal_panel):
    panel, _ = signal_panel
    config = GPConfig(population_size=80, generations=12, max_depth=5, max_nodes=30, seed=2)
    gp = GP(config, panel)
    gp.run()

    best = [h["best_fitness"] for h in gp.history]
    mean = [h["mean_fitness"] for h in gp.history]
    # elitism makes best fitness non-decreasing across generations
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(best, best[1:], strict=False))
    # the search actually learned the signal and lifted the population
    assert gp.history[-1]["best_ic"] > 0.3
    assert mean[-1] > mean[0]
    # converged: the best plateaus by the end
    assert abs(best[-1] - best[-2]) < 0.1


def test_crossover_type_safe(signal_panel):
    panel, _ = signal_panel
    gp = GP(GPConfig(max_depth=6, max_nodes=40, seed=3), panel)
    for _ in range(10_000):
        a = gp.generator.generate(grow=True)
        b = gp.generator.generate(grow=False)
        child = gp._crossover(a, b)
        assert is_valid(child)
        assert child.depth() <= 6 and child.size() <= 40


def test_checkpoint_resume(signal_panel, tmp_path):
    panel, _ = signal_panel
    config = GPConfig(population_size=30, generations=3, max_depth=5, max_nodes=25, seed=5)

    straight = GP(config, panel)
    straight.run()  # 3 generations in one go

    ckpt = tmp_path / "ckpt.json"
    partial = GP(config, panel)
    partial.run(generations=2, checkpoint_path=ckpt)  # 2 generations, checkpointed
    resumed = GP.from_checkpoint(ckpt, panel)
    resumed.run()  # resumes to config.generations (3)

    assert [to_json(i.tree) for i in straight.population] == [
        to_json(i.tree) for i in resumed.population
    ]
    assert [i.fitness for i in straight.population] == [i.fitness for i in resumed.population]
    assert straight.history == resumed.history


def test_workers_do_not_change_search_trajectory(signal_panel):
    panel, _ = signal_panel
    config = GPConfig(population_size=24, generations=2, max_depth=5, max_nodes=25, seed=19)

    serial = GP(config, panel, workers=1)
    parallel = GP(config, panel, workers=4)
    serial.run()
    parallel.run()

    assert [to_json(i.tree) for i in serial.population] == [
        to_json(i.tree) for i in parallel.population
    ]
    assert [i.fitness for i in serial.population] == [i.fitness for i in parallel.population]
    assert serial.history == parallel.history
    assert serial.trial_count == parallel.trial_count


def test_cancelled_generation_is_transactional(signal_panel, monkeypatch):
    import alphalineage.core.gp as gp_module

    panel, _ = signal_panel
    gp = GP(GPConfig(population_size=12, generations=2, seed=23), panel, workers=2)
    gp.initialize()
    population_before = [to_json(i.tree) for i in gp.population]
    cache_before = dict(gp._cache)
    trials_before = gp.trial_count

    made = 0

    def offspring():
        nonlocal made
        made += 1
        tree = Node(
            "add_scalar", (Node("close"), Node("const", value=float(1000 + made)))
        )
        return tree, [0], "test"

    monkeypatch.setattr(gp, "_offspring", offspring)
    monkeypatch.setattr(
        gp_module,
        "score_trees",
        lambda trees, *args, **kwargs: [(0.1, {"ic": 0.1, "ic_ir": 0.0}) for _ in trees],
    )
    polls = 0

    def stop() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 2

    with pytest.raises(TrainingCancelled):
        gp._step(stop=stop)

    assert gp.generation == 0
    assert [to_json(i.tree) for i in gp.population] == population_before
    assert gp._cache == cache_before
    assert gp.trial_count == trials_before


def test_seeded_init_contains_seeds_and_marks_lineage(signal_panel):
    from alphalineage.library.store import LineageStore

    panel, _ = signal_panel
    seeds = [
        Node("rank", (Node("ts_mean", (Node("close"), Node("window", value=5))),)),
        Node("rank", (Node("volume"),)),
    ]
    store = LineageStore()
    gp = GP(
        GPConfig(population_size=20, generations=1, max_depth=5, max_nodes=25, seed=7),
        panel,
        recorder=store,
    )
    gp.initialize(seeds)

    assert [to_json(i.tree) for i in gp.population[:2]] == [to_json(s) for s in seeds]
    gen0 = [n for n in store.nodes if n.generation == 0]
    assert [n.op for n in gen0[:2]] == ["seed", "seed"]
    assert all(n.op == "init" for n in gen0[2:])


def test_seed_rejected_with_named_offender(signal_panel):
    panel, _ = signal_panel
    gp = GP(GPConfig(population_size=10, max_depth=2, max_nodes=5, seed=0), panel)

    with pytest.raises(ValueError, match="bogus"):
        gp.initialize([Node("bogus")])
    # a WINDOW-typed root cannot seed a SIGNAL population
    with pytest.raises(ValueError, match="window"):
        gp.initialize([Node("window", value=5)])
    # depth/size limits hold for seeds too
    deep = Node("rank", (Node("ts_mean", (Node("close"), Node("window", value=5))),))
    with pytest.raises(ValueError, match="depth"):
        gp.initialize([deep])


def test_seed_limits_apply_after_user_formula_expansion(signal_panel):
    from alphalineage.core.extensions import ARG, register_operator, unregister_operator

    panel, _ = signal_panel
    register_operator(
        "expanded_seed_limit_test",
        [DType.SERIES],
        DType.SERIES,
        Node(
            "add",
            (
                Node(ARG, value=0),
                Node("mul_scalar", (Node(ARG, value=0), Node("const", value=0.0))),
            ),
        ),
    )
    try:
        compact = Node("expanded_seed_limit_test", (Node("close"),))
        assert compact.size() == 2
        gp = GP(GPConfig(population_size=6, max_depth=6, max_nodes=2, seed=0), panel)
        with pytest.raises(ValueError, match="expanded expression size"):
            gp.initialize([compact])
    finally:
        unregister_operator("expanded_seed_limit_test")


def test_legacy_macro_checkpoint_rescores_with_expanded_parsimony(signal_panel, tmp_path):
    import json

    from alphalineage.core.extensions import ARG, register_operator, unregister_operator
    from alphalineage.core.fitness import forward_returns, score_tree

    panel, _ = signal_panel
    name = "expanded_checkpoint_penalty_test"
    register_operator(
        name,
        [DType.SERIES],
        DType.SERIES,
        Node(
            "add",
            (
                Node(ARG, value=0),
                Node("mul_scalar", (Node(ARG, value=0), Node("const", value=0.0))),
            ),
        ),
    )
    try:
        config = GPConfig(
            population_size=6,
            generations=1,
            max_depth=6,
            max_nodes=10,
            parsimony=0.1,
            seed=72,
        )
        macro = Node(name, (Node("volume"),))
        original = GP(config, panel)
        original.initialize([macro])
        checkpoint = tmp_path / "legacy-macro.json"
        original.save_checkpoint(checkpoint)
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        state["scorer_version"] = SCORER_VERSION - 1
        state["population"][0]["fitness"] = 123.0
        checkpoint.write_text(json.dumps(state), encoding="utf-8")

        resumed = GP.from_checkpoint(checkpoint, panel)
        previous_trials = resumed.trial_count
        assert resumed._requires_rescore
        resumed.run(generations=resumed.generation)
        assert not resumed._requires_rescore
        assert resumed.trial_count == previous_trials
        restored = next(individual for individual in resumed.population if individual.tree == macro)
        expected = score_tree(
            macro,
            panel,
            forward_returns(panel),
            parsimony=config.parsimony,
            min_names=config.min_names,
        )
        assert restored.fitness == pytest.approx(expected[0], abs=1e-12)
    finally:
        unregister_operator(name)


def test_legacy_horizon_checkpoint_rescores_against_cumulative_target(signal_panel, tmp_path):
    import json

    from alphalineage.core.fitness import score_tree

    panel, _ = signal_panel
    horizon = 3
    config = GPConfig(
        population_size=6,
        generations=1,
        max_depth=4,
        max_nodes=12,
        horizon=horizon,
        parsimony=0.0,
        seed=73,
    )
    seed = Node("rank", (Node("volume"),))
    original = GP(config, panel)
    original.initialize([seed])
    checkpoint = tmp_path / "legacy-horizon.json"
    original.save_checkpoint(checkpoint)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    state["scorer_version"] = SCORER_VERSION - 1
    state["population"][0]["fitness"] = 123.0
    checkpoint.write_text(json.dumps(state), encoding="utf-8")

    resumed = GP.from_checkpoint(checkpoint, panel)
    previous_trials = resumed.trial_count
    assert resumed._requires_rescore
    resumed.run(generations=resumed.generation)

    cumulative_target = panel["close"].shift(-horizon).div(panel["close"]).sub(1.0)
    expected = score_tree(
        seed,
        panel,
        cumulative_target,
        parsimony=config.parsimony,
        min_names=config.min_names,
    )
    legacy_offset_target = panel["returns"].shift(-horizon)
    legacy = score_tree(
        seed,
        panel,
        legacy_offset_target,
        parsimony=config.parsimony,
        min_names=config.min_names,
    )
    restored = next(individual for individual in resumed.population if individual.tree == seed)

    assert not resumed._requires_rescore
    assert resumed.trial_count == previous_trials
    assert expected[0] != pytest.approx(legacy[0], abs=1e-6)
    assert restored.fitness == pytest.approx(expected[0], abs=1e-12)
    assert restored.metrics == pytest.approx(expected[1], abs=1e-12)


def test_checkpoint_carries_trial_count(signal_panel, tmp_path):
    panel, _ = signal_panel
    config = GPConfig(population_size=20, generations=3, max_depth=4, max_nodes=20, seed=5)
    ckpt = tmp_path / "ckpt.json"

    partial = GP(config, panel)
    partial.run(generations=2, checkpoint_path=ckpt)
    counted = partial.trial_count
    assert counted > 0

    resumed = GP.from_checkpoint(ckpt, panel)
    assert resumed.trial_count == counted  # never silently resets to 0
    assert not resumed._requires_rescore
    assert resumed._cache  # current-population scores are restored as resume-time cache hits
    resumed.run()
    assert resumed.trial_count >= counted

    import json

    saved = json.loads(ckpt.read_text(encoding="utf-8"))
    assert saved["scorer_version"] == SCORER_VERSION
    assert saved["evolution_version"] == EVOLUTION_VERSION
    assert saved["scorer_backend"] == partial.scorer_backend


def test_checkpoint_rescores_when_numerical_backend_changes(signal_panel, tmp_path):
    panel, _ = signal_panel
    config = GPConfig(population_size=12, generations=1, max_depth=3, max_nodes=12, seed=51)
    checkpoint = tmp_path / "backend-change.json"
    original = GP(config, panel)
    original.run(checkpoint_path=checkpoint)

    import json

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    state["scorer_backend"] = "different-kernel"
    checkpoint.write_text(json.dumps(state), encoding="utf-8")

    resumed = GP.from_checkpoint(checkpoint, panel)
    previous_trials = resumed.trial_count
    assert resumed._requires_rescore
    assert not resumed._cache

    resumed.run(generations=resumed.generation)
    assert not resumed._requires_rescore
    assert resumed.trial_count == previous_trials


def test_stop_callback_halts_early(signal_panel):
    panel, _ = signal_panel
    config = GPConfig(population_size=20, generations=50, max_depth=4, max_nodes=20, seed=6)
    gp = GP(config, panel)
    gp.run(stop=lambda: gp.generation >= 2)
    assert gp.generation == 2


def test_rescore_on_new_panel_changes_fitness_not_trees(signal_panel, noise_panel):
    panel, _ = signal_panel
    config = GPConfig(population_size=20, generations=2, max_depth=4, max_nodes=20, seed=8)
    gp = GP(config, panel)
    gp.run()
    trees_before = [to_json(i.tree) for i in gp.population]
    fits_before = [i.fitness for i in gp.population]
    trials_before = gp.trial_count

    gp.panel = noise_panel
    gp.rescore_population()

    assert [to_json(i.tree) for i in gp.population] == trees_before
    assert [i.fitness for i in gp.population] != fits_before
    assert gp.trial_count >= trials_before  # counted trials never shrink


# --- supporting -----------------------------------------------------------------
def test_tournament_selects_best(signal_panel):
    panel, _ = signal_panel
    gp = GP(GPConfig(population_size=20, tournament_size=20, seed=0), panel)
    gp.initialize()
    # a full-population tournament must return the single best individual (and its index)
    best, idx = gp._tournament()
    assert best.fitness == max(i.fitness for i in gp.population)
    assert gp.population[idx] is best


def test_point_mutation_tweaks_constant_and_stays_valid(signal_panel):
    panel, _ = signal_panel
    gp = GP(GPConfig(seed=0), panel)
    tree = Node("ts_mean", (Node("close"), Node("window", value=5)))
    seen_change = False
    for _ in range(50):
        mutated = gp._point_mutation(tree)
        assert is_valid(mutated)
        if to_json(mutated) != to_json(tree):
            seen_change = True
    assert seen_change


def test_subtree_mutation_is_type_correct(signal_panel):
    panel, _ = signal_panel
    gp = GP(GPConfig(max_depth=6, max_nodes=40, seed=0), panel)
    tree = Node("add", (Node("close"), Node("volume")))
    for _ in range(200):
        mutated = gp._subtree_mutation(tree)
        assert is_valid(mutated)
        assert mutated.depth() <= 6 and mutated.size() <= 40


def test_replace_at_and_positions_round_trip():
    tree = Node("add", (Node("close"), Node("volume")))
    positions = iter_positions(tree, DType.SIGNAL)
    # root + two leaves
    assert len(positions) == 3
    # replacing the root's left child
    rebuilt = replace_at(tree, (0,), Node("returns"))
    assert rebuilt == Node("add", (Node("returns"), Node("volume")))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"population_size": 0}, "population_size"),
        ({"generations": 0}, "generations"),
        ({"population_size": 2, "tournament_size": 3}, "tournament_size"),
        ({"crossover_rate": 1.1}, "crossover_rate"),
        ({"subtree_mutation_rate": float("nan")}, "subtree_mutation_rate"),
        ({"min_depth": 7, "max_depth": 6}, "min_depth"),
        ({"elitism": -1}, "elitism"),
        ({"ic_method": "kendall"}, "ic_method"),
        ({"horizon": 0}, "horizon"),
        ({"time_budget_s": -1.0}, "time_budget_s"),
    ],
)
def test_gp_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GPConfig(**kwargs)


def test_gp_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown GP config field.*populaton_size"):
        GPConfig.from_dict({"populaton_size": 20})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"population_size": MAX_POPULATION_SIZE + 1}, "population_size"),
        ({"generations": MAX_GENERATIONS + 1}, "generations"),
        ({"max_depth": MAX_TREE_DEPTH + 1}, "max_depth"),
        ({"max_nodes": MAX_TREE_NODES + 1}, "max_nodes"),
        ({"min_names": MAX_MIN_NAMES + 1}, "min_names"),
        ({"horizon": MAX_HORIZON + 1}, "horizon"),
        ({"time_budget_s": MAX_TIME_BUDGET_S + 1}, "time_budget_s"),
        ({"population_size": 2_000, "generations": 501}, "must not exceed"),
    ],
)
def test_gp_config_enforces_resource_ceilings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GPConfig(**kwargs)
