"""Training-only novelty, with frozen references and bounded on-disk rank panels.

The evaluator hands us each already-computed candidate. No target/validation data is accepted.
Reference expressions and representative order are checkpointed; scratch rank arrays are rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import tempfile
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from alphalineage.core import cpp
from alphalineage.core.evaluate import evaluate
from alphalineage.core.extensions import expand_all
from alphalineage.core.simplify import simplify
from alphalineage.core.tree import Node, from_dict, to_dict, to_json
from alphalineage.data.paths import temporary_dir

NOVELTY_VERSION = 1
# Full panels are an expendable acceleration cache, not search state. Keep enough
# free space for screening arrays and the atomically replaced recovery checkpoint.
RANK_CACHE_BYTES = 8 * 1024**3
SCRATCH_RESERVE_BYTES = 1024**3
logger = logging.getLogger(__name__)


def _cleanup_storage(directory, mappings):
    # The finalizer owns the directory and mappings, but never the context itself. This
    # also keeps TemporaryDirectory's finalizer from running before its mappings close.
    for mapping in list(mappings.values()):
        try:
            if not mapping.closed:
                mapping.close()
        except (OSError, BufferError):
            logger.warning("Could not close a novelty mapping in %s", directory.name, exc_info=True)
    mappings.clear()
    try:
        directory.cleanup()
    except OSError:
        logger.warning("Novelty scratch remains at %s after cleanup", directory.name, exc_info=True)


def standardized(values, min_names):
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    centered = np.where(
        finite, values - np.nansum(values, axis=1)[:, None] / np.maximum(1, count[:, None]), 0.0
    )
    norm = np.sqrt((centered * centered).sum(axis=1))
    valid = (count >= max(3, min_names)) & (norm > 0)
    return (
        (centered / np.where(valid, norm, 1)[:, None]).astype(np.float32),
        valid,
        bool(np.all((count == 0) | (count == values.shape[1]))),
    )


def identity(tree: Node) -> str:
    return to_json(simplify(expand_all(tree), preserve_missing=True))


def multiplier(correlation: float) -> float:
    return 1 - 0.5 * min(1.0, max(0.0, (correlation - 0.7) / 0.3))


def ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(
        np.where(np.isfinite(values), values, np.nan), axis=1, method="average", nan_policy="omit"
    )


def correlations(left: np.ndarray, right: np.ndarray, min_names: int) -> tuple[float, int]:
    native = cpp.ranked_correlations(left, right, min_names)
    if native is not None:
        return native
    shared = np.isfinite(left) & np.isfinite(right)
    # Pairwise ranks must be recomputed when the shared set differs.
    count = shared.sum(axis=1)
    # Whole missing dates need no reranking; only differing name masks do.
    changed = (count > 0) & (
        (count != np.isfinite(left).sum(axis=1)) | (count != np.isfinite(right).sum(axis=1))
    )
    x, y = left, right
    if changed.any():
        x, y = left.copy(), right.copy()
        x[changed] = ranks(np.where(shared[changed], left[changed], np.nan))
        y[changed] = ranks(np.where(shared[changed], right[changed], np.nan))
    x = np.where(shared, x - np.nansum(x, axis=1)[:, None] / np.maximum(1, count[:, None]), 0.0)
    y = np.where(shared, y - np.nansum(y, axis=1)[:, None] / np.maximum(1, count[:, None]), 0.0)
    scale = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    valid = (count >= max(3, min_names)) & (scale > 0)
    n = int(valid.sum())
    return (float(((x * y).sum(axis=1)[valid] / scale[valid]).mean()) if n > 0 else 0.0, n)


class NoveltyContext:
    def __init__(
        self,
        panel,
        references: list[dict[str, Any]],
        *,
        min_names: int,
        memory_budget_bytes: int | None = None,
        families: list[dict[str, Any]] | None = None,
        progress=None,
        rank_cache_budget_bytes: int = RANK_CACHE_BYTES,
    ):
        if (
            isinstance(rank_cache_budget_bytes, bool)
            or not isinstance(rank_cache_budget_bytes, int)
            or rank_cache_budget_bytes < 0
        ):
            raise ValueError("rank_cache_budget_bytes must be a nonnegative integer")
        self.panel, self.references, self.min_names = panel, references, min_names
        self.fingerprint = hashlib.sha256(
            json.dumps(references, sort_keys=True).encode()
        ).hexdigest()
        self.sample = np.unique(
            np.linspace(0, len(panel.dates) - 1, min(128, len(panel.dates)), dtype=int)
        )
        self.directory = tempfile.TemporaryDirectory(
            prefix="alphalineage-novelty-", dir=temporary_dir()
        )
        self._mappings = {}
        self._closed = False
        self._finalizer = weakref.finalize(self, _cleanup_storage, self.directory, self._mappings)
        self.memory_budget = max(1, (memory_budget_bytes or 256 * 1024 * 1024) // 4)
        self.items: list[dict[str, Any]] = []
        self.families: list[dict[str, Any]] = []
        self.unavailable: list[str] = []
        self.blocks = []
        self.block_size = 16
        self.rank_cache_budget = int(rank_cache_budget_bytes)
        self._rank_cache = OrderedDict()
        self._rank_cache_bytes = 0
        self._pinned_ranks = {}
        try:
            total = len(references) + len(families or [])
            if progress is not None:
                progress(0, total)
            for index, reference in enumerate(references):
                try:
                    self._add(reference, known=True)
                except (ValueError, KeyError, TypeError, FloatingPointError):
                    self.unavailable.append(reference["key"])
                if progress is not None:
                    progress(index + 1, total)
            for index, item in enumerate(families or []):
                self._add(item, known=False)
                self.families.append(item)
                if progress is not None:
                    progress(len(references) + index + 1, total)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._finalizer()
            self.items.clear()
            self.blocks.clear()
            self._rank_cache.clear()
            self._rank_cache_bytes = 0

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Novelty context is closed")

    def _own(self, array):
        if isinstance(array, np.memmap):
            self._mappings[id(array._mmap)] = array._mmap
        return array

    def _release(self, array):
        if isinstance(array, np.memmap):
            mapping = array._mmap
            if not mapping.closed:
                mapping.close()
            self._mappings.pop(id(mapping), None)

    def _save_array(self, path, values):
        self._reserve_scratch(values.nbytes + 256)
        try:
            np.save(path, values, allow_pickle=False)
        except OSError as exc:
            raise OSError(f"Cannot write novelty scratch at {path}: {exc}") from exc

    def _add(
        self,
        spec: dict[str, Any],
        *,
        known: bool,
        values: np.ndarray | None = None,
        already_ranked: bool = False,
        screened: tuple | None = None,
    ):
        self._ensure_open()
        tree = from_dict(spec["tree"])
        key = identity(tree)
        for index, item in enumerate(self.items):
            if item["identity"] == key:
                return index
        if values is None:
            evaluated = evaluate(tree, self.panel)
            if not isinstance(evaluated, pd.DataFrame):
                raise ValueError("Reference does not produce a panel")
            values = evaluated.to_numpy(dtype=float)
        path = Path(self.directory.name) / f"{len(self.items)}.ranks.npy"
        # Retain the exact screening ranks and expression, not a full raw panel for
        # every trial. Full confirmations can rebuild ranks from the training panel.
        sample = (
            screened[0]
            if screened is not None
            else values[self.sample]
            if already_ranked
            else ranks(values[self.sample])
        ).astype(np.float32)
        # Samples stay mapped as well when the configured memory share is exhausted.
        if sum(item["sample"].nbytes for item in self.items) + sample.nbytes > self.memory_budget:
            sample_path = path.with_name(f"{len(self.items)}.sample.npy")
            self._save_array(sample_path, sample)
            sample = self._own(np.load(sample_path, mmap_mode="r"))
        unit, valid, whole_rows = (
            screened[1:] if screened is not None else standardized(sample, self.min_names)
        )
        index = len(self.items)
        block_index, slot = divmod(index, self.block_size)
        if block_index == len(self.blocks):
            shape = (self.block_size, sample.size)
            if (block_index + 1) * math.prod(shape) * 4 < self.memory_budget // 2:
                block = np.zeros(shape, dtype=np.float32)
            else:
                self._reserve_scratch(math.prod(shape) * 4 + 256)
                block = self._own(
                    np.lib.format.open_memmap(
                        Path(self.directory.name) / f"block-{block_index}.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=shape,
                    )
                )
            self.blocks.append(block)
        self.blocks[block_index][slot] = unit.ravel()
        self.items.append(
            {
                "valid": valid,
                "whole_rows": whole_rows,
                "spec": spec,
                "identity": key,
                "known": known,
                "path": path,
                "sample": sample,
            }
        )
        return index

    def _discard_cached_ranks(self, path):
        # Call only after mapped readers close, including on Windows.
        path.unlink(missing_ok=True)
        self._rank_cache_bytes -= self._rank_cache.pop(path, 0)

    def _evict_oldest(self):
        victim = next((p for p in self._rank_cache if p not in self._pinned_ranks), None)
        if victim is None:
            return False
        try:
            self._discard_cached_ranks(victim)
        except OSError:
            logger.warning("Cannot evict novelty ranks at %s", victim, exc_info=True)
            return False
        return True

    def _reserve_scratch(self, required):
        # Growing screening arrays take priority over optional full-panel caches.
        while self._rank_cache:
            try:
                free = shutil.disk_usage(self.directory.name).free
            except OSError:
                return
            if free >= required + SCRATCH_RESERVE_BYTES or not self._evict_oldest():
                return

    def _make_cache_room(self, required):
        if required > self.rank_cache_budget:
            return False
        try:
            free = shutil.disk_usage(self.directory.name).free
        except OSError:
            return False
        capacity = min(
            self.rank_cache_budget,
            max(0, self._rank_cache_bytes + free - SCRATCH_RESERVE_BYTES),
        )
        while self._rank_cache_bytes + required > capacity:
            if not self._evict_oldest():
                return False
        return True

    @staticmethod
    def _fill_ranks(destination, values):
        # Rank the original float64 values before narrowing the ordinal ranks, so
        # close values never turn into artificial ties when rebuilding an eviction.
        for offset in range(0, len(values), 128):
            destination[offset : offset + 128] = ranks(values[offset : offset + 128])

    @contextmanager
    def _full_ranks(self, item):
        self._ensure_open()
        path = item["path"]
        ranked = None
        pinned = False
        try:
            if path in self._rank_cache:
                ranked = self._own(np.load(path, mmap_mode="r"))
                self._rank_cache.move_to_end(path)
            if ranked is None:
                evaluated = evaluate(from_dict(item["spec"]["tree"]), self.panel)
                if not isinstance(evaluated, pd.DataFrame):
                    raise ValueError("Reference does not produce a panel")
                values = evaluated.to_numpy(dtype=float)
                if self._make_cache_room(values.size * 4 + 256):
                    try:
                        ranked = self._own(
                            np.lib.format.open_memmap(
                                path, mode="w+", dtype=np.float32, shape=values.shape
                            )
                        )
                        self._fill_ranks(ranked, values)
                        ranked.flush()
                        size = path.stat().st_size
                        self._rank_cache[path] = size
                        self._rank_cache_bytes += size
                    except OSError:
                        # Caching is optional. A storage failure must not lose a
                        # generation when a single transient panel still fits RAM.
                        self._release(ranked)
                        ranked = None
                        self._discard_cached_ranks(path)
                        logger.warning("Cannot cache novelty ranks at %s; using RAM", path)
                if ranked is None:
                    ranked = np.empty(values.shape, dtype=np.float32)
                    self._fill_ranks(ranked, values)
                del values, evaluated
            if isinstance(ranked, np.memmap):
                self._pinned_ranks[path] = self._pinned_ranks.get(path, 0) + 1
                pinned = True
            yield ranked
        finally:
            self._release(ranked)
            if pinned:
                self._pinned_ranks[path] -= 1
                if not self._pinned_ranks[path]:
                    del self._pinned_ranks[path]
            if path not in self._rank_cache:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Cannot remove incomplete novelty ranks at %s", path)

    def state(self):
        return {
            "version": NOVELTY_VERSION,
            "references": self.references,
            "min_names": self.min_names,
            "fingerprint": self.fingerprint,
            "families": list(self.families),
            "unavailable": list(self.unavailable),
        }

    def mark(self):
        return len(self.items), len(self.families)

    def rollback(self, mark):
        for item in self.items[mark[0] :]:
            self._release(item["sample"])
            try:
                self._discard_cached_ranks(item["path"])
                if isinstance(item["sample"], np.memmap):
                    Path(item["sample"].filename).unlink(missing_ok=True)
            except OSError:
                logger.warning("Cannot remove rolled-back novelty scratch", exc_info=True)
                self._rank_cache_bytes -= self._rank_cache.pop(item["path"], 0)
        keep_blocks = (mark[0] + self.block_size - 1) // self.block_size
        for block in self.blocks[keep_blocks:]:
            self._release(block)
        del self.blocks[keep_blocks:]
        del self.items[mark[0] :]
        del self.families[mark[1] :]

    def apply(self, tree: Node, values: np.ndarray, scored):
        self._ensure_open()
        fitness, metrics = scored
        metrics = dict(metrics)
        candidate = None
        small = ranks(values[self.sample])
        unit, candidate_valid, whole_rows = standardized(small, self.min_names)
        screened = []
        measured = 0
        for block_index, block in enumerate(self.blocks):
            start = block_index * self.block_size
            count = min(self.block_size, len(self.items) - start)
            if count <= 0:
                break
            # einsum avoids oversubscribing BLAS threads inside the training worker budget.
            dots = np.einsum("ij,j->i", block[:count], unit.ravel(), optimize=False)
            for offset, dot in enumerate(dots):
                index = start + offset
                item = self.items[index]
                if whole_rows and item["whole_rows"]:
                    dates = int((candidate_valid & item["valid"]).sum())
                    corr = float(dot) / max(1, dates)
                else:
                    corr, dates = correlations(small, item["sample"], self.min_names)
                enough = dates >= max(5, int(len(self.sample) * 0.6))
                measured += int(enough and item["known"])
                # Float32 screening has a small guard band; confirmations use exact ranks.
                if enough and abs(corr) >= 0.59999:
                    screened.append((abs(corr), index))
        known_rho, family = 0.0, -1
        # Confirm known references and evolutionary representatives independently so the
        # latter cannot crowd the former out of the economic-overlap penalty.
        confirmed = []
        for known in (True, False):
            choices = [(corr, i) for corr, i in screened if self.items[i]["known"] == known]
            for _, i in sorted(choices, key=lambda item: (-item[0], item[1]))[:3]:
                with self._full_ranks(self.items[i]) as other:
                    if candidate is None:
                        candidate = ranks(values)
                    # Stream confirmations without retaining mapped panels between calls.
                    total, count = 0.0, 0
                    for offset in range(0, len(candidate), 128):
                        corr, n = correlations(
                            candidate[offset : offset + 128],
                            other[offset : offset + 128],
                            self.min_names,
                        )
                        if n > 0:
                            total += corr * n
                            count += n
                if count >= max(5, int(len(candidate) * 0.6)):
                    rho = abs(total / count)
                    if known:
                        known_rho = max(known_rho, rho)
                    if rho >= 0.98:
                        confirmed.append(i)
        if confirmed:
            family = min(confirmed)
        else:
            family = len(self.items)
            spec = {
                "key": "candidate:" + hashlib.sha256(identity(tree).encode()).hexdigest(),
                "tree": to_dict(simplify(expand_all(tree), preserve_missing=True)),
            }
            # Screening prototypes grow with the search; full panels use a bounded cache.
            family = self._add(
                spec,
                known=False,
                values=values if candidate is None else candidate,
                already_ranked=candidate is not None,
                screened=(small, unit, candidate_valid, whole_rows),
            )
            if (
                not any(item["key"] == spec["key"] for item in self.families)
                and not self.items[family]["known"]
            ):
                self.families.append(spec)
        scale = multiplier(known_rho)
        raw = metrics.get("raw_objective", metrics.get("ic", 0.0))
        penalty = metrics.get("complexity_penalty", raw - fitness)
        metrics.update(
            novelty_multiplier=scale,
            novelty_penalty=raw * (1 - scale),
            novelty_correlation=known_rho,
            novelty_measured=float(measured),
            novelty_family=float(family),
        )
        return raw * scale - penalty, metrics
