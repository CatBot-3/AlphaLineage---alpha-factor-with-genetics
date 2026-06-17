"""Phase 5 acceptance + supporting tests for the factor library (tests/test_library.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalineage.core.evaluate import evaluate
from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.tree import Node, to_json
from alphalineage.library.combine import combine
from alphalineage.library.diversity import prune, signal_correlation
from alphalineage.library.store import LineageStore
from alphalineage.validation.deflated_sharpe import deflated_sharpe_ratio, sharpe_ratio
from alphalineage.validation.performance import long_short_returns


def test_lineage_replay(signal_panel, tmp_path):
    panel, _ = signal_panel
    store = LineageStore()
    gp = GP(
        GPConfig(population_size=20, generations=3, max_depth=4, max_nodes=20, seed=1),
        panel,
        recorder=store,
    )
    gp.run()

    gens = store.generations()
    assert len(gens) == 4  # generations 0..3
    # the last recorded generation is exactly the final population
    assert [to_json(t) for t in gens[-1]] == [to_json(i.tree) for i in gp.population]

    # save -> load reconstructs every generation losslessly
    loaded = LineageStore.load(store.save(tmp_path / "lineage.json"))
    assert [[to_json(t) for t in g] for g in loaded.generations()] == [
        [to_json(t) for t in g] for g in gens
    ]

    # parent integrity: non-init nodes reference existing, earlier ids
    ids = {n.id for n in loaded.nodes}
    for node in loaded.nodes:
        if node.op == "init":
            assert node.parents == []
        else:
            assert all(p in ids and p < node.id for p in node.parents)


def test_lineage_records_fitness(signal_panel, tmp_path):
    panel, _ = signal_panel
    store = LineageStore()
    gp = GP(
        GPConfig(population_size=15, generations=2, max_depth=4, max_nodes=20, seed=2),
        panel,
        recorder=store,
    )
    gp.run()

    assert all(isinstance(n.fitness, float) for n in store.nodes)
    # the final generation's recorded fitnesses match the population's
    final = [n for n in store.nodes if n.generation == gp.generation]
    assert [n.fitness for n in final] == [i.fitness for i in gp.population]

    # round-trips through JSON; files without fitness load as None
    loaded = LineageStore.load(store.save(tmp_path / "lineage.json"))
    assert [n.fitness for n in loaded.nodes] == [n.fitness for n in store.nodes]
    legacy = store.to_dict()
    for node in legacy["nodes"]:
        node.pop("fitness", None)
    import json

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert all(n.fitness is None for n in LineageStore.load(legacy_path).nodes)


def test_diversity_threshold(synthetic_panel):
    panel = synthetic_panel
    trees = [
        Node("close"),
        Node("mul_scalar", (Node("close"), Node("const", value=2.0))),  # ~+1 corr with close
        Node("neg", (Node("close"),)),  # -1 corr with close
        Node("volume"),
        Node("returns"),
        Node("high"),  # ~+1 corr with close
    ]
    kept = prune(trees, panel, fitness=lambda t: float(t.size()), threshold=0.7)

    assert len(kept) < len(trees)  # correlated/duplicate factors were dropped
    signals = [evaluate(t, panel) for t in kept]
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            assert abs(signal_correlation(signals[i], signals[j])) < 0.7


def test_combo_beats_best_single():
    rng = np.random.default_rng(0)
    n_days, n_syms, k = 120, 12, 5
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    syms = [f"S{i}" for i in range(n_syms)]

    def z(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    # k independent latent factors; forward returns are their (noisy) sum -> each predicts weakly.
    latents = [
        pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=dates, columns=syms)
        for _ in range(k)
    ]
    signal = latents[0].copy()
    for extra in latents[1:]:
        signal = signal.add(z(extra), fill_value=0.0)
    noise = pd.DataFrame(rng.normal(0, 1, (n_days, n_syms)), index=dates, columns=syms)
    fwd = 0.02 * z(signal) + 0.12 * noise
    train = dates[: int(n_days * 0.7)]

    combo = combine(latents, fwd, train_dates=train)
    returns = [long_short_returns(s, fwd) for s in latents]
    var_sr = max(float(np.var([sharpe_ratio(r) for r in returns], ddof=1)), 0.02)

    # The best single factor is the winner of a large search -> deflate by the trial count.
    # The mega-alpha is one principled construction of the diverse set -> deflate by 1.
    best_single = max(deflated_sharpe_ratio(r, n_trials=200, var_sr=var_sr) for r in returns)
    combo_dsr = deflated_sharpe_ratio(long_short_returns(combo, fwd), n_trials=1, var_sr=var_sr)
    assert combo_dsr > best_single


def test_signal_correlation_bounds_and_symmetry(synthetic_panel):
    a = evaluate(Node("close"), synthetic_panel)
    b = evaluate(Node("volume"), synthetic_panel)
    assert -1.0 <= signal_correlation(a, b) <= 1.0
    assert np.isclose(signal_correlation(a, b), signal_correlation(b, a))
    assert np.isclose(signal_correlation(a, a), 1.0)
