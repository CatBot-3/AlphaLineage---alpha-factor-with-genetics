"""Training novelty never accepts validation labels and survives transactions/resume."""

import json

import numpy as np
import pytest

from alphalineage.core.extensions import expand_all
from alphalineage.core.gp import GP, GPConfig, TrainingCancelled
from alphalineage.core.novelty import NoveltyContext, correlations, multiplier, ranks
from alphalineage.core.tree import Node, to_dict, to_json


def references():
    return [{"key": "reference:volume:r1", "tree": to_dict(Node("volume")), "revision": 1}]


def config(**kwargs):
    return GPConfig(
        population_size=16,
        generations=2,
        max_depth=4,
        max_nodes=20,
        novelty_mode="balanced",
        seed=4,
        **kwargs,
    )


def test_penalty_inversion_masks_and_novel_composition(signal_panel):
    panel, _ = signal_panel
    novelty = NoveltyContext(panel, references(), min_names=5)
    values = panel["volume"].to_numpy()
    fitness, metrics = novelty.apply(
        Node("volume"), values, (0.2, {"raw_objective": 0.21, "complexity_penalty": 0.01})
    )
    assert fitness == pytest.approx(0.095)
    assert metrics["novelty_correlation"] == pytest.approx(1)
    _, inverse = novelty.apply(
        Node("neg", (Node("volume"),)),
        -values,
        (0.2, {"raw_objective": 0.21, "complexity_penalty": 0.01}),
    )
    assert inverse["novelty_family"] == metrics["novelty_family"]
    assert inverse["novelty_multiplier"] == pytest.approx(0.5)
    novel = np.random.default_rng(2).normal(size=values.shape)
    _, independent = novelty.apply(
        Node("close"), novel, (0.2, {"raw_objective": 0.21, "complexity_penalty": 0.01})
    )
    assert independent["novelty_multiplier"] == 1
    missing = values.copy()
    missing[:, :9] = np.nan
    _, insufficient = novelty.apply(
        Node("high"), missing, (0.2, {"raw_objective": 0.21, "complexity_penalty": 0.01})
    )
    assert insufficient["novelty_measured"] == 0
    assert multiplier(0.7) == 1 and multiplier(0.85) == pytest.approx(0.75)


def test_pairwise_masks_rerank_only_the_intersection():
    left = np.tile([10, 20, 30, 40, np.nan], (12, 1))
    right = np.tile([np.nan, 2, 3, 4, 5], (12, 1))
    correlation, count = correlations(ranks(left), ranks(right), 3)
    assert correlation == pytest.approx(1)
    assert count == 12


def test_native_masked_rank_correlation_matches_python(monkeypatch):
    from alphalineage.core import cpp

    if not cpp.available() or not hasattr(cpp._EXT, "ranked_correlations"):
        pytest.skip("native novelty kernel is optional")
    rng = np.random.default_rng(812)
    left = rng.integers(0, 5, (128, 100)).astype(float)
    right = rng.integers(0, 10, left.shape).astype(float)
    left[rng.random(left.shape) < 0.2] = np.nan
    right[rng.random(right.shape) < 0.3] = np.nan
    left[0] = np.nan
    right[1] = 1
    x, y = ranks(left), ranks(right)
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    expected = correlations(x, y, 5)
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    actual = correlations(x, y, 5)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert actual[1] == expected[1]


def test_native_observer_receives_the_same_outputs(synthetic_panel):
    from alphalineage.core.evaluate import evaluate
    from alphalineage.core.fitness import forward_returns, score_trees

    trees = [Node("volume"), Node("neg", (Node("close"),))]
    seen = []

    def observer(tree, values, score):
        np.testing.assert_allclose(values, evaluate(tree, synthetic_panel), equal_nan=True)
        seen.append(tree)
        return score

    expected = score_trees(trees, synthetic_panel, forward_returns(synthetic_panel), workers=2)
    actual = score_trees(
        trees, synthetic_panel, forward_returns(synthetic_panel), workers=2, observer=observer
    )
    assert actual == expected and seen == trees


def test_families_shortlist_but_keep_all_label_scored_trials(signal_panel):
    panel, _ = signal_panel
    gp = GP(config(), panel, novelty_state={"references": references()})
    trees = [Node("volume"), Node("neg", (Node("volume"),)), Node("close")]
    gp._individuals(trees, phase="initializing")
    assert gp.trial_count == 3
    assert len(gp.validation_candidates()) == 2
    gp._individuals(trees, phase="initializing")
    assert gp.trial_count == 3 and gp.structural_skips == 3


def test_cancel_rolls_back_novelty_and_scores(signal_panel, monkeypatch):
    panel, _ = signal_panel
    gp = GP(config(), panel, novelty_state={"references": references()})
    monkeypatch.setattr(gp, "_scoring_chunk_size", lambda _: 1)
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 2

    before = gp.novelty.state()
    with pytest.raises(TrainingCancelled):
        gp._individuals([Node("close"), Node("high")], phase="initializing", stop=stop)
    assert gp.trial_count == 0 and gp.structural_skips == 0
    assert gp.novelty.state() == before


def test_checkpoint_resume_and_worker_parity(signal_panel, tmp_path):
    panel, _ = signal_panel
    settings = config()
    straight = GP(settings, panel, workers=1, novelty_state={"references": references()})
    straight.run()
    partial = GP(settings, panel, workers=2, novelty_state={"references": references()})
    checkpoint = tmp_path / "novelty.json"
    partial.run(generations=1, checkpoint_path=checkpoint)
    resumed = GP.from_checkpoint(checkpoint, panel, workers=2)
    resumed.run()
    assert [to_json(i.tree) for i in straight.population] == [
        to_json(i.tree) for i in resumed.population
    ]
    assert [i.fitness for i in straight.population] == [i.fitness for i in resumed.population]
    assert straight.history == resumed.history
    assert straight.novelty.state() == resumed.novelty.state()
    assert json.loads(checkpoint.read_text())["novelty"]["references"] == references()


def test_reference_revision_changes_fingerprint(synthetic_panel):
    one = NoveltyContext(synthetic_panel, references(), min_names=5)
    changed = [{**references()[0], "revision": 2}]
    two = NoveltyContext(synthetic_panel, changed, min_names=5)
    assert one.fingerprint != two.fingerprint


def test_expanded_macro_evaluates_once(signal_panel):
    from alphalineage.core.extensions import clear_user_operators, ensure_operator
    from alphalineage.core.types import DType

    ensure_operator(
        "novelty_volume", [], DType.SIGNAL, {"name": "rank", "children": [{"name": "volume"}]}
    )
    try:
        tree = Node("novelty_volume")
        gp = GP(config(), signal_panel[0])
        gp._individuals([tree, expand_all(tree)], phase="initializing")
        assert gp.trial_count == 1 and gp.structural_skips == 1
    finally:
        clear_user_operators()


def test_family_pruning_keeps_lineage_aligned(signal_panel):
    from alphalineage.library.store import LineageStore

    store = LineageStore()
    gp = GP(config(), signal_panel[0], recorder=store, novelty_state={"references": references()})
    seeds = [Node("mul_scalar", (Node("volume"), Node("const", value=i + 1))) for i in range(12)]
    gp.initialize(seeds=seeds)
    gp.run()
    assert len(store._current_gen_ids) == len(gp.population)
    assert all(parent < item.id for item in store.nodes for parent in item.parents)


def test_future_panels_cannot_change_training_novelty(signal_panel):
    from alphalineage.core.panel import Panel

    panel, _ = signal_panel
    cutoff = panel.dates[100]
    changed = Panel(
        {
            name: frame.where(frame.index.to_series() <= cutoff, -frame * 9, axis=0)
            for name, frame in panel.fields.items()
        }
    )

    def fingerprint(source):
        training = Panel({name: frame.loc[:cutoff] for name, frame in source.fields.items()})
        gp = GP(config(), training, novelty_state={"references": references()})
        gp.run()
        return gp.novelty.state(), gp.history, gp._cache

    assert fingerprint(panel) == fingerprint(changed)


def test_structural_identity_preserves_missing_and_nonfinite_semantics(signal_panel):
    from alphalineage.core.novelty import identity

    close = Node("close")
    assert identity(close) != identity(Node("ts_ema", (close, Node("window", value=1))))
    assert identity(close) != identity(Node("signed_power", (close, Node("const", value=1))))
    assert identity(close) == identity(Node("neg", (Node("neg", (close,)),)))


def test_cached_predictions_keep_expression_specific_complexity(signal_panel):
    from alphalineage.core.extensions import clear_user_operators, ensure_operator
    from alphalineage.core.types import DType

    ensure_operator(
        "novelty_noop",
        [],
        DType.SIGNAL,
        {"name": "mul_scalar", "children": [{"name": "volume"}, {"name": "const", "value": 1}]},
    )
    try:
        gp = GP(config(parsimony=0.01), signal_panel[0])
        macro, plain = gp._individuals([Node("novelty_noop"), Node("volume")], phase="initializing")
        assert gp.trial_count == 1
        assert macro.metrics["ic"] == plain.metrics["ic"]
        assert macro.metrics["expanded_complexity"] == 3
        assert plain.metrics["expanded_complexity"] == 1
        assert plain.fitness - macro.fitness == pytest.approx(0.02)
        assert gp.validation_candidates()[0].tree == Node("volume")
    finally:
        clear_user_operators()
