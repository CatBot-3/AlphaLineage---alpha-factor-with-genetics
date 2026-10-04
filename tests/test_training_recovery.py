"""Windows mapping ownership and atomic committed-generation recovery."""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.novelty import NoveltyContext, ranks
from alphalineage.core.tree import Node, to_dict
from alphalineage.data import identifiers, paths


def references():
    return [{"key": "volume", "tree": to_dict(Node("volume"))}]


def settings():
    return GPConfig(
        population_size=12,
        generations=2,
        max_depth=3,
        max_nodes=12,
        novelty_mode="balanced",
        seed=11,
    )


def test_temporary_storage_defaults_to_data_and_supports_override(tmp_path, monkeypatch):
    assert paths.temporary_dir() == (paths.data_dir() / "tmp").resolve()
    override = tmp_path / "scratch"
    monkeypatch.setenv("ALPHALINEAGE_TEMP_DIR", str(override))
    assert paths.temporary_dir() == override
    obstacle = tmp_path / "file"
    obstacle.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("ALPHALINEAGE_TEMP_DIR", str(obstacle / "scratch"))
    with pytest.raises(OSError, match="Cannot create AlphaLineage temporary storage"):
        paths.temporary_dir()


def test_owned_mappings_close_before_directory_removal(signal_panel):
    with NoveltyContext(
        signal_panel[0], references(), min_names=5, memory_budget_bytes=4
    ) as context:
        directory = Path(context.directory.name)
        sample, block = context.items[0]["sample"], context.blocks[0]
        assert isinstance(sample, np.memmap) and isinstance(block, np.memmap)
        with context._full_ranks(context.items[0]) as ranked:
            rank_mapping = ranked._mmap
            assert not rank_mapping.closed
        assert rank_mapping.closed
        assert len(context._mappings) == 2
    assert not directory.exists()
    assert sample._mmap.closed and block._mmap.closed
    context.close()


def test_garbage_collection_closes_mappings_without_unraisable_error(signal_panel, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "unraisablehook", errors.append)
    context = NoveltyContext(signal_panel[0], references(), min_names=5, memory_budget_bytes=4)
    directory = Path(context.directory.name)
    del context
    gc.collect()
    assert not errors and not directory.exists()


def test_failed_reference_preparation_cleans_up_and_keeps_original_error(signal_panel):
    def fail_after_first(done, total):
        if done:
            raise RuntimeError("preparation interrupted")

    with pytest.raises(RuntimeError, match="preparation interrupted"):
        NoveltyContext(
            signal_panel[0],
            references(),
            min_names=5,
            memory_budget_bytes=4,
            progress=fail_after_first,
        )
    assert list(paths.temporary_dir().iterdir()) == []


def test_rank_failure_releases_transient_mappings(signal_panel, monkeypatch):
    import alphalineage.core.novelty as novelty

    with NoveltyContext(
        signal_panel[0], references(), min_names=5, memory_budget_bytes=4
    ) as context:

        def fail(values):
            raise RuntimeError("rank failure")

        monkeypatch.setattr(novelty, "ranks", fail)
        with pytest.raises(RuntimeError, match="rank failure"):
            with context._full_ranks(context.items[0]):
                pass
        assert len(context._mappings) == 2


def test_rollback_releases_sample_before_reusing_filename(signal_panel):
    with NoveltyContext(
        signal_panel[0], references(), min_names=5, memory_budget_bytes=4
    ) as context:
        mark = context.mark()
        spec = {"key": "close", "tree": to_dict(Node("close"))}
        context._add(spec, known=False)
        removed_mapping = context.items[-1]["sample"]._mmap
        context.rollback(mark)
        assert removed_mapping.closed
        context._add(spec, known=False)
        assert len(context.items) == 2


def test_checkpoint_write_failure_preserves_previous_complete_file(
    signal_panel, tmp_path, monkeypatch
):
    path = tmp_path / "checkpoint.json"
    with GP(settings(), signal_panel[0]) as engine:
        engine.initialize()
        engine.save_checkpoint(path)
        original = path.read_bytes()

        def fail_replace(source, target):
            raise OSError("disk unavailable")

        monkeypatch.setattr(identifiers.os, "replace", fail_replace)
        with pytest.raises(OSError, match="disk unavailable"):
            engine.run(generations=1, checkpoint_path=path)
        assert path.read_bytes() == original
        assert json.loads(original)["generation"] == 0
        assert not list(tmp_path.glob(".*.tmp"))


def test_failed_checkpoint_restore_releases_novelty_storage(signal_panel, tmp_path):
    path = tmp_path / "checkpoint.json"
    with GP(settings(), signal_panel[0]) as engine:
        engine.initialize()
        engine.save_checkpoint(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("rng_state")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeyError, match="rng_state"):
        GP.from_checkpoint(path, signal_panel[0])
    assert list(paths.temporary_dir().iterdir()) == []


def test_rank_cache_eviction_rebuilds_exact_ranks_and_closes_windows_maps(signal_panel):
    panel = signal_panel[0]
    specs = [{"key": name, "tree": to_dict(Node(name))} for name in ("close", "high", "volume")]
    panel_bytes = len(panel.dates) * len(panel.symbols) * 4
    with NoveltyContext(
        panel,
        specs,
        min_names=5,
        memory_budget_bytes=4,
        rank_cache_budget_bytes=panel_bytes + 256,
    ) as context:
        # Preparation retains only samples/blocks, including when restoring families.
        assert not any(item["path"].exists() for item in context.items)
        retained_maps = []
        for index in (0, 1, 2, 0):
            with context._full_ranks(context.items[index]) as ranked:
                np.testing.assert_array_equal(ranked, ranks(panel[specs[index]["key"]].to_numpy()))
                retained_maps.append(ranked._mmap)
            assert all(mapping.closed for mapping in retained_maps)
            assert len(list(Path(context.directory.name).glob("*.ranks.npy"))) == 1
            assert context._rank_cache_bytes <= context.rank_cache_budget
        assert not context.items[1]["path"].exists()


def test_pinned_rank_cache_is_not_evicted_by_nested_comparison(signal_panel):
    panel = signal_panel[0]
    specs = references() + [{"key": "close", "tree": to_dict(Node("close"))}]
    panel_bytes = len(panel.dates) * len(panel.symbols) * 4
    with NoveltyContext(
        panel, specs, min_names=5, rank_cache_budget_bytes=panel_bytes + 256
    ) as context:
        with context._full_ranks(context.items[0]) as pinned:
            with context._full_ranks(context.items[1]) as other:
                assert not isinstance(other, np.memmap)
                assert not pinned._mmap.closed
                np.testing.assert_array_equal(pinned, ranks(panel["volume"].to_numpy()))
        assert pinned._mmap.closed


def test_low_disk_space_discards_optional_cache_and_uses_ram(signal_panel, monkeypatch):
    import alphalineage.core.novelty as novelty

    with NoveltyContext(signal_panel[0], references(), min_names=5) as context:
        with context._full_ranks(context.items[0]):
            pass
        monkeypatch.setattr(
            novelty.shutil, "disk_usage", lambda path: novelty.shutil._ntuple_diskusage(0, 0, 0)
        )
        assert not context._make_cache_room(1000)
        assert context._rank_cache_bytes == 0
        monkeypatch.undo()
        with context._full_ranks(context.items[0]):
            pass
        monkeypatch.setattr(
            novelty.shutil, "disk_usage", lambda path: novelty.shutil._ntuple_diskusage(0, 0, 0)
        )
        context._reserve_scratch(1000)
        assert context._rank_cache_bytes == 0
        with context._full_ranks(context.items[0]) as ranked:
            assert not isinstance(ranked, np.memmap)
            np.testing.assert_array_equal(ranked, ranks(signal_panel[0]["volume"].to_numpy()))
        assert not context.items[0]["path"].exists()


def test_rank_cache_write_failure_does_not_abort_training(signal_panel, monkeypatch):
    with NoveltyContext(signal_panel[0], references(), min_names=5) as context:

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(np.lib.format, "open_memmap", fail)
        with context._full_ranks(context.items[0]) as ranked:
            assert not isinstance(ranked, np.memmap)
            np.testing.assert_array_equal(ranked, ranks(signal_panel[0]["volume"].to_numpy()))
        assert context._rank_cache_bytes == 0


def test_evicted_ranks_preserve_float64_ties_and_missing_values(signal_panel):
    from alphalineage.core.panel import Panel

    panel = signal_panel[0]
    frame = panel["close"].copy()
    frame.iloc[:, :] = 1.0
    frame.iloc[:, 1] = np.nextafter(1.0, 2.0)
    frame.iloc[0, 0] = np.nan
    precise = Panel({**panel.fields, "close": frame})
    with NoveltyContext(
        precise,
        [{"key": "close", "tree": to_dict(Node("close"))}],
        min_names=5,
        rank_cache_budget_bytes=0,
    ) as context:
        with context._full_ranks(context.items[0]) as ranked:
            np.testing.assert_array_equal(ranked, ranks(frame.to_numpy()))
            assert ranked[1, 1] > ranked[1, 0]


def test_bounded_cache_and_resumption_preserve_search_and_lineage(signal_panel, tmp_path):
    from alphalineage.library.store import LineageStore

    panel = signal_panel[0]
    stores = [LineageStore() for _ in range(3)]
    snapshot = tmp_path / "bounded.json"
    with GP(
        settings(), panel, recorder=stores[0], novelty_state={"references": references()}
    ) as full:
        full.run()
        expected = full.checkpoint_state()
    with GP(
        settings(), panel, recorder=stores[1], novelty_state={"references": references()}
    ) as partial:
        partial.novelty.rank_cache_budget = len(panel.dates) * len(panel.symbols) * 4 + 256
        partial.run(generations=1, checkpoint_path=snapshot)
        saved_lineage = stores[1].to_dict()
    stores[2] = LineageStore.from_dict(saved_lineage)
    stores[2].continue_from(1)
    with GP.from_checkpoint(snapshot, panel, recorder=stores[2]) as resumed:
        resumed.novelty.rank_cache_budget = 0
        resumed.run()
        actual = resumed.checkpoint_state()
    for key in (
        "generation",
        "rng_state",
        "trials",
        "history",
        "score_cache",
        "population",
        "novelty",
    ):
        assert actual[key] == expected[key], key
    assert stores[0].to_dict()["nodes"] == stores[2].to_dict()["nodes"]
