"""Phase 2 acceptance + supporting tests for the GP loop (tests/test_gp.py)."""

from __future__ import annotations

from alphaforge.core.evaluate import evaluate
from alphaforge.core.fitness import mean_ic
from alphaforge.core.gp import GP, GPConfig, iter_positions, replace_at
from alphaforge.core.tree import Node, is_valid, to_json
from alphaforge.core.types import DType


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


# --- supporting -----------------------------------------------------------------
def test_tournament_selects_best(signal_panel):
    panel, _ = signal_panel
    gp = GP(GPConfig(population_size=20, tournament_size=20, seed=0), panel)
    gp.initialize()
    # a full-population tournament must return the single best individual
    assert gp._tournament().fitness == max(i.fitness for i in gp.population)


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
