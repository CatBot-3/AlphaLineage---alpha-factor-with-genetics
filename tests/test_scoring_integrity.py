"""Regression tests for price semantics, directional scoring, and robust reporting."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.engine import backtest
from alphalineage.backtest.portfolio import RankProportional
from alphalineage.backtest.reporting import backtest_report
from alphalineage.core import cpp
from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import daily_ic, score_tree
from alphalineage.core.gp import GP, MAX_CANONICAL_COPIES, GPConfig, Individual
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node
from alphalineage.data import schema
from alphalineage.data.adjust import adjust
from alphalineage.data.cache import ParquetCache, merge_price_frames
from alphalineage.data.integrity import audit_price_frame
from alphalineage.data.tiingo_client import _to_frame as tiingo_frame
from alphalineage.data.yfinance_provider import _to_frame as yfinance_frame
from alphalineage.validation.selection import select_validation_candidate


def _split_history(*, adjusted: bool) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    close = [100.0, 102.0, 51.0, 52.0] if not adjusted else [50.0, 51.0, 51.0, 52.0]
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [100.0, 100.0, 200.0, 200.0],
            "Dividends": 0.0,
            "Stock Splits": [0.0, 0.0, 2.0, 0.0],
        },
        index=dates,
    )


def test_yahoo_split_adjusted_and_tiingo_raw_normalize_to_same_close():
    yahoo = yfinance_frame(_split_history(adjusted=True))
    rows = []
    for date, row in _split_history(adjusted=False).iterrows():
        rows.append(
            {
                "date": date.isoformat(),
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row["Volume"],
                "divCash": 0.0,
                "splitFactor": row["Stock Splits"] or 1.0,
            }
        )
    tiingo = tiingo_frame(rows)

    assert schema.resolve_price_basis(yahoo) == "split_adjusted"
    assert schema.resolve_price_basis(tiingo) == "raw"
    pd.testing.assert_series_equal(
        adjust(yahoo)["adj_close"],
        adjust(tiingo)["adj_close"],
        check_names=False,
    )


def test_cache_persists_basis_and_rejects_mixed_incremental_sources(tmp_path):
    cache = ParquetCache(tmp_path)
    yahoo = yfinance_frame(_split_history(adjusted=True))
    cache.store("TEST", yahoo)
    assert cache.metadata("TEST") == {
        "price_basis": "split_adjusted",
        "provider": "yfinance",
        "metadata_version": 1,
        "legacy_inferred": False,
    }

    raw = schema.with_price_metadata(
        schema.normalize(
            yahoo.rename(
                columns={
                    "open": "open",
                }
            )
        ),
        price_basis="raw",
        provider="tiingo",
    )
    with pytest.raises(schema.PriceBasisError, match="different adjustment bases"):
        merge_price_frames([yahoo, raw])


def test_legacy_split_basis_inference_and_residual_action_audit():
    raw = schema.normalize(
        yfinance_frame(_split_history(adjusted=False)).drop(
            columns=[], errors="ignore"
        )
    )
    raw.attrs.clear()
    adjusted = schema.normalize(yfinance_frame(_split_history(adjusted=True)))
    adjusted.attrs.clear()
    assert schema.resolve_price_basis(raw) == "raw"
    assert schema.resolve_price_basis(adjusted) == "split_adjusted"
    assert not audit_price_frame(raw)
    assert not audit_price_frame(adjusted)


def _alternating_panel() -> tuple[Panel, pd.DataFrame]:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    columns = list("ABC")
    factor = pd.DataFrame(
        np.tile([1.0, 2.0, 3.0], (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    target = factor.copy()
    target.iloc[1::2] = target.iloc[1::2, ::-1].to_numpy()
    panel = Panel.from_prices(
        open=factor,
        high=factor,
        low=factor,
        close=factor,
        volume=factor * 10.0,
    )
    return panel, target


def test_fitness_uses_absolute_aggregate_not_mean_daily_absolute():
    panel, target = _alternating_panel()
    fitness, metrics = score_tree(
        Node("close"),
        panel,
        target,
        absolute=True,
        min_names=3,
    )
    assert fitness == pytest.approx(0.0, abs=1e-15)
    assert metrics["signed_ic"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["oriented_ic"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["mean_abs_ic"] == pytest.approx(1.0)
    assert metrics["sign_consistency"] == pytest.approx(0.5)


def test_negative_training_polarity_is_persisted_through_validation():
    panel, _ = _alternating_panel()
    target = -panel["close"]
    train_fitness, train_metrics = score_tree(
        Node("close"),
        panel,
        target,
        min_names=3,
    )
    candidate = Individual(Node("close"), train_fitness, train_metrics)
    selection = select_validation_candidate(
        [candidate],
        panel,
        target,
        panel.dates,
        min_names=3,
        validation_folds=3,
        fold_embargo=0,
        min_valid_dates_per_fold=2,
    )
    assert selection.validated
    assert selection.polarity == -1
    oriented = evaluate(selection.oriented_tree, panel)
    assert isinstance(oriented, pd.DataFrame)
    assert daily_ic(oriented, target, min_names=3).mean() == pytest.approx(1.0)


def test_validation_uses_three_embargo_separated_folds_and_majority_direction():
    dates = pd.date_range("2018-01-01", periods=190, freq="B")
    symbols = [f"S{index}" for index in range(6)]
    values = np.tile(np.arange(1.0, 7.0), (len(dates), 1))
    close = pd.DataFrame(values, index=dates, columns=symbols)
    panel = Panel.from_prices(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=close * 10.0,
    )
    forward = close.copy()
    # With 190 dates, 3 folds, and two 5-date embargoes, every fold has exactly
    # 60 observations. The last fold reverses direction, leaving a 2/3 majority.
    forward.loc[dates[130:]] *= -1.0
    candidate = Individual(
        Node("close"),
        1.0,
        {"signed_ic": 1.0, "ic": 1.0},
    )
    selection = select_validation_candidate(
        [candidate],
        panel,
        forward,
        dates,
        min_names=5,
        validation_folds=3,
        fold_embargo=5,
        min_valid_dates_per_fold=60,
        complexity_penalty_mode="normalized_budget",
        complexity_penalty_value=0.005,
        max_nodes=40,
    )

    assert selection.validated is True
    assert [fold["valid_dates"] for fold in selection.fold_metrics] == [60, 60, 60]
    assert [fold["positive"] for fold in selection.fold_metrics] == [True, True, False]
    assert selection.positive_folds == 2
    assert selection.validation_metrics["median_oriented_ic"] == pytest.approx(1.0)
    assert selection.validation_metrics["worst_fold_ic"] == pytest.approx(-1.0)
    assert selection.complexity_deduction == pytest.approx(0.005 / 40.0)


def test_validation_rejects_candidate_with_only_one_positive_fold():
    dates = pd.date_range("2018-01-01", periods=190, freq="B")
    symbols = [f"S{index}" for index in range(6)]
    close = pd.DataFrame(
        np.tile(np.arange(1.0, 7.0), (len(dates), 1)),
        index=dates,
        columns=symbols,
    )
    panel = Panel.from_prices(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=close,
    )
    forward = -close
    forward.loc[dates[130:]] *= -1.0
    selection = select_validation_candidate(
        [Individual(Node("close"), 1.0, {"signed_ic": 1.0})],
        panel,
        forward,
        dates,
        min_names=5,
        validation_folds=3,
        fold_embargo=5,
        min_valid_dates_per_fold=60,
    )
    assert selection.validated is False
    assert selection.positive_folds == 1
    assert "at least 2 positive" in str(selection.reason)


class _NonNeutralWeightingScheme:
    name = "nonneutral"

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        weights = factor * 0.0
        weights.iloc[:, 0] = 1.0
        return weights


def test_validation_rejects_weighting_scheme_that_breaks_portfolio_contract():
    panel, target = _alternating_panel()

    with pytest.raises(ValueError, match="must be dollar neutral"):
        select_validation_candidate(
            [Individual(Node("close"), 1.0, {"signed_ic": 1.0})],
            panel,
            target,
            panel.dates,
            min_names=3,
            validation_folds=3,
            fold_embargo=0,
            min_valid_dates_per_fold=1,
            weighting_scheme=_NonNeutralWeightingScheme(),
        )


def test_validation_rejects_sparse_varying_and_portfolio_exposure_below_sixty_percent():
    dates = pd.date_range("2018-01-01", periods=300, freq="B")
    symbols = [f"S{index}" for index in range(6)]
    values = np.ones((len(dates), len(symbols)), dtype="float64")
    varied = np.arange(1.0, 7.0)
    for fold_start in (0, 100, 200):
        values[fold_start : fold_start + 59] = varied
    close = pd.DataFrame(values, index=dates, columns=symbols)
    panel = Panel.from_prices(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=close,
    )
    selection = select_validation_candidate(
        [Individual(Node("close"), 1.0, {"signed_ic": 1.0})],
        panel,
        close,
        dates,
        min_names=5,
        validation_folds=3,
        fold_embargo=0,
        min_valid_dates_per_fold=10,
        minimum_fold_coverage=0.60,
    )

    assert selection.validated is False
    for fold in selection.fold_metrics:
        assert fold["varying_factor_coverage"] == pytest.approx(0.59)
        assert fold["exposure_coverage"] == pytest.approx(0.59)
        assert "low_varying_factor_coverage" in fold["coverage_failures"]
        assert "low_exposure_coverage" in fold["coverage_failures"]


def test_population_copy_limit_and_diversity_history_are_worker_invariant(
    signal_panel, monkeypatch
):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    panel, _ = signal_panel
    config = GPConfig(
        population_size=24,
        generations=2,
        max_depth=4,
        max_nodes=18,
        seed=41,
    )
    runs = []
    for workers in (1, 4):
        gp = GP(config, panel, workers=workers)
        gp.run()
        counts = Counter(gp._canonical_key(ind.tree) for ind in gp.population)
        assert max(counts.values()) <= MAX_CANONICAL_COPIES
        assert gp.history[-1]["unique_tree_ratio"] >= 0.5
        runs.append(gp)
    assert [ind.tree for ind in runs[0].population] == [
        ind.tree for ind in runs[1].population
    ]
    assert runs[0].history == runs[1].history


def test_checkpoint_retains_all_first_seen_validation_candidates(
    signal_panel, tmp_path, monkeypatch
):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "python")
    panel, _ = signal_panel
    gp = GP(
        GPConfig(
            population_size=20,
            generations=2,
            max_depth=4,
            max_nodes=18,
            seed=73,
        ),
        panel,
    )
    checkpoint = tmp_path / "checkpoint.json"
    gp.run(checkpoint_path=checkpoint)
    before = gp.searched_individuals()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(payload["score_cache"]) == len(before)
    assert len(before) > len(gp.population)

    restored = GP.from_checkpoint(checkpoint, panel)
    after = restored.searched_individuals()
    assert [item.tree for item in after] == [item.tree for item in before]
    assert [item.fitness for item in after] == [item.fitness for item in before]
    assert [item.metrics for item in after] == [item.metrics for item in before]
    assert restored.trial_count == gp.trial_count


def test_missing_return_and_insolvency_paths_are_not_compounded():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    columns = list("ABC")
    close = pd.DataFrame(
        np.tile([1.0, 2.0, 3.0], (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    returns = pd.DataFrame(0.0, index=dates, columns=columns)
    returns.iloc[1, 2] = -3.0
    panel = Panel(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": close,
            "vwap": close,
            "returns": returns,
        }
    )
    factor = close.copy()
    result = backtest(
        factor,
        panel,
        returns.shift(-1),
        RankProportional(),
        TransactionCostModel(0.0, 0.0),
    )
    assert result.insolvent
    assert not result.usable

    report = backtest_report(
        factor,
        panel,
        returns.shift(-1),
        RankProportional(),
        TransactionCostModel(0.0, 0.0),
        dates,
        min_names=3,
    )
    assert report["integrity"]["equity_terminated_reason"] == "insolvent"
    assert report["normalized_equity"][-1]["value"] == 0.0

    missing = Panel({name: value.copy() for name, value in panel.fields.items()})
    missing["returns"].iloc[1, 2] = np.nan
    missing_result = backtest(
        factor,
        missing,
        returns.shift(-1),
        RankProportional(),
        TransactionCostModel(0.0, 0.0),
    )
    assert missing_result.missing_return_observations == 1
    assert not missing_result.usable


@pytest.mark.skipif(not cpp.supports_native_scoring("spearman"), reason="native scorer unavailable")
def test_native_directional_metrics_match_python(signal_panel, monkeypatch):
    monkeypatch.setenv("ALPHALINEAGE_EVALUATOR", "auto")
    panel, _ = signal_panel
    target = panel["returns"].shift(-1)
    trees = [Node("close"), Node("volume"), Node("returns")]
    expected = [score_tree(tree, panel, target) for tree in trees]
    actual = cpp.score_many(trees, panel, target, workers=3)
    for native, reference in zip(actual, expected, strict=True):
        assert native is not None
        assert native[0] == pytest.approx(reference[0], abs=1e-12)
        assert native[1] == pytest.approx(reference[1], abs=1e-12)
