"""P11-T2 - measuring an expression tree.

What survives from the old anatomy tests is everything that checks *arithmetic*. The aspect-label
tests are gone with the hand-written dictionary they exercised; labels now come from the model.

The single most important property here is the compounded lookback. It is the number a language
model reliably gets wrong by eye, which is exactly why it stayed computed.
"""

from __future__ import annotations

import pytest

from alphalineage.core.tree import Node
from alphalineage.explain.anatomy import (
    base_name,
    effective_lookback,
    formula_text,
    measure,
    render_markdown,
)


def series(name: str) -> Node:
    return Node(name)


def window(value: int) -> Node:
    return Node("window", value=value)


# --- rendering --------------------------------------------------------------------
def test_formula_text_does_not_need_the_registry() -> None:
    """Stored factors may reference macros that are not registered right now."""
    tree = Node("some_unregistered_macro__r7", (series("close"), window(20)))
    assert formula_text(tree) == "some_unregistered_macro__r7(close, 20)"


def test_base_name_strips_a_catalog_revision() -> None:
    assert base_name("ta_atr__r2") == "ta_atr"
    assert base_name("ts_mean") == "ts_mean"


# --- the lookback arithmetic ------------------------------------------------------
def test_a_flat_rolling_window_consumes_one_less_than_its_length() -> None:
    measured = measure(Node("ts_std", (series("close"), window(20))))
    assert measured.windows.effective_lookback_bars == 19


def test_nested_windows_compound() -> None:
    """The number a model gets wrong by eye: 20 and 20 nested is 38, not 20."""
    nested = Node("ts_std", (Node("ts_mean", (series("close"), window(20))), window(20)))
    assert measure(nested).windows.effective_lookback_bars == 38


def test_a_deep_stack_reaches_far_further_than_its_largest_window() -> None:
    tree = series("close")
    for _ in range(4):
        tree = Node("ts_mean", (tree, window(30)))
    measurement = measure(tree)
    assert measurement.windows.maximum == 30
    assert measurement.windows.effective_lookback_bars == 116  # 4 x 29


def test_delay_consumes_its_full_window_unlike_a_rolling_mean() -> None:
    shift = measure(Node("delay", (series("close"), window(20))))
    rolling = measure(Node("ts_sum", (series("close"), window(20))))
    assert shift.windows.effective_lookback_bars == 20
    assert rolling.windows.effective_lookback_bars == 19


def test_the_deepest_path_wins_not_the_sum_of_all_branches() -> None:
    tree = Node(
        "add",
        (
            Node("ts_mean", (series("close"), window(10))),
            Node("ts_std", (Node("ts_mean", (series("close"), window(50))), window(50))),
        ),
    )
    assert measure(tree).windows.effective_lookback_bars == 98


def test_cumsum_marks_the_lookback_unbounded() -> None:
    measurement = measure(Node("ts_cumsum", (series("close"),)))
    assert measurement.windows.unbounded_lookback
    assert "unbounded_lookback" in {d.code for d in measurement.diagnostics}


def test_recursive_smoothing_is_flagged() -> None:
    assert measure(Node("ts_ema", (series("close"), window(20)))).windows.recursive_smoothing


def test_effective_lookback_is_exposed_directly() -> None:
    bars, unbounded, recursive = effective_lookback(Node("ts_mean", (series("close"), window(10))))
    assert (bars, unbounded, recursive) == (9, False, False)


# --- window profile ---------------------------------------------------------------
def test_window_profile_bands_and_statistics() -> None:
    tree = Node(
        "add",
        (
            Node("ts_mean", (series("close"), window(3))),
            Node(
                "add",
                (
                    Node("ts_mean", (series("close"), window(10))),
                    Node("ts_mean", (series("close"), window(60))),
                ),
            ),
        ),
    )
    profile = measure(tree).windows
    assert profile.values == (3, 10, 60)
    assert (profile.minimum, profile.median, profile.maximum) == (3, 10.0, 60)
    assert (profile.short, profile.medium, profile.long) == (1, 1, 1)


def test_an_expression_with_no_windows_reports_none() -> None:
    profile = measure(Node("rank", (series("close"),))).windows
    assert profile.count == 0 and profile.effective_lookback_bars == 0


# --- structure --------------------------------------------------------------------
def test_structure_reports_root_normalization_and_repeats() -> None:
    inner = Node("ts_mean", (series("close"), window(20)))
    structure = measure(Node("rank", (Node("sub", (inner, inner)),))).structure
    assert structure.root == "rank"
    assert structure.root_is_cross_sectional and structure.scale_invariant
    assert structure.data_fields == ("close",)
    assert any(item["occurrences"] == 2 for item in structure.repeated_subtrees)


def test_indicators_are_read_from_the_surface_tree() -> None:
    surface = Node("ta_atr__r2", (window(14),))
    expanded = Node("ts_rma", (Node("sub", (series("high"), series("low"))), window(14)))
    measurement = measure(surface, expanded=expanded)
    assert measurement.structure.indicators == ("ta_atr",)
    assert measurement.structure.data_fields == ("high", "low")


def test_operator_counts_need_no_dictionary_to_maintain() -> None:
    """The point of keeping the measuring: a brand-new operator counts without any code change."""
    tree = Node("rank", (Node("ts_corr", (series("close"), series("volume"), window(20))),))
    counts = measure(tree).structure.operator_counts
    assert counts == {"rank": 1, "ts_corr": 1}


# --- units ------------------------------------------------------------------------
def test_unit_mismatch_flags_adding_price_to_volume() -> None:
    measurement = measure(Node("add", (series("close"), series("volume"))))
    assert measurement.units.mismatches
    mismatch = measurement.units.mismatches[0]
    assert {mismatch["left_unit"], mismatch["right_unit"]} == {"price", "volume"}
    assert "unit_mismatch" in {d.code for d in measurement.diagnostics}


def test_matching_units_do_not_trigger_a_mismatch() -> None:
    measurement = measure(Node("sub", (series("high"), series("low"))))
    assert not measurement.units.mismatches
    assert measurement.units.output_unit == "price"


def test_ratio_of_like_units_is_dimensionless() -> None:
    assert measure(Node("div", (series("close"), series("open")))).units.output_unit == (
        "dimensionless"
    )


# --- diagnostics ------------------------------------------------------------------
def test_missing_cross_sectional_normalization_is_flagged() -> None:
    plain = measure(Node("ts_mean", (series("close"), window(5))))
    assert "no_cross_sectional" in {d.code for d in plain.diagnostics}
    ranked = measure(Node("rank", (Node("ts_mean", (series("close"), window(5))),)))
    assert "no_cross_sectional" not in {d.code for d in ranked.diagnostics}


def test_long_lookback_against_a_short_horizon_is_flagged() -> None:
    tree = Node("ts_mean", (series("close"), window(120)))
    assert "long_lookback_vs_horizon" in {d.code for d in measure(tree, horizon=1).diagnostics}
    assert "long_lookback_vs_horizon" not in {d.code for d in measure(tree, horizon=60).diagnostics}


def test_lookback_consuming_the_training_window_is_a_warning() -> None:
    tree = Node("ts_mean", (series("close"), window(120)))
    diagnostics = {d.code: d for d in measure(tree, train_bars=500).diagnostics}
    assert diagnostics["lookback_vs_train_window"].severity == "warning"


def test_nested_smoothing_chain_is_flagged() -> None:
    tree = Node(
        "ts_mean",
        (Node("ts_ema", (Node("ts_rma", (series("close"), window(5))), window(5))), window(5)),
    )
    assert "nested_smoothing" in {d.code for d in measure(tree).diagnostics}


def test_parameter_pinned_at_a_policy_bound_is_reported() -> None:
    policy = {"inputs": [{"name": "length", "tuning": {"min": 5, "max": 100, "default": 20}}]}
    at_bound = measure(
        Node("ta_sma__r2", (window(5),)),
        expanded=Node("ts_mean", (series("close"), window(5))),
        policy_bounds={"ta_sma__r2": policy},
    )
    assert "parameter_at_policy_bound" in {d.code for d in at_bound.diagnostics}

    interior = measure(
        Node("ta_sma__r2", (window(20),)),
        expanded=Node("ts_mean", (series("close"), window(20))),
        policy_bounds={"ta_sma__r2": policy},
    )
    assert "parameter_at_policy_bound" not in {d.code for d in interior.diagnostics}


# --- serialization ----------------------------------------------------------------
def test_to_dict_is_json_shaped_and_carries_no_labels() -> None:
    payload = measure(Node("rank", (Node("ts_std", (series("returns"), window(20))),))).to_dict()
    assert payload["windows"]["effective_lookback_bars"] == 19
    assert set(payload) == {
        "formula",
        "expanded_formula",
        "horizon",
        "windows",
        "structure",
        "units",
        "diagnostics",
    }
    # The taxonomy is gone; nothing here claims to know what the factor *means*.
    assert "aspects" not in payload


def test_measurement_is_deterministic() -> None:
    tree = Node("rank", (Node("ts_corr", (series("close"), series("volume"), window(20))),))
    assert measure(tree).to_dict() == measure(tree).to_dict()


def test_markdown_render_covers_every_section() -> None:
    text = render_markdown(measure(Node("ts_mean", (series("close"), window(20))), horizon=5))
    for heading in ("Lookback", "Structure", "Units", "Diagnostics"):
        assert heading in text
    assert "nested windows compound" in text


def test_the_aspect_taxonomy_is_gone() -> None:
    """It needed a human edit per operator. Deleting it is the point of P11-T2."""
    with pytest.raises(ImportError):
        import alphalineage.explain.aspects  # noqa: F401
