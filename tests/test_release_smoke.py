from alphalineage.smoke import release_smoke


def test_release_smoke_sees_native_evaluator_and_packaged_catalogs() -> None:
    result = release_smoke()

    assert result["native_evaluator"] is True
    assert result["native_abi"] >= 6
    assert result["bundled_universes"] == {
        "builtin-sp500-current": 503,
        "builtin-djia-current": 30,
        "builtin-nasdaq100-current": 103,
    }
    assert result["classification_subsets"] == 31
    assert result["starter_formulas"] == 96
    assert result["active_starter_formulas"] == 95
