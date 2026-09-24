"""Training-only novelty, with frozen references and bounded on-disk rank panels.

The evaluator hands us each already-computed candidate. No target/validation data is accepted.
Reference expressions and representative order are checkpointed; scratch rank arrays are rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
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

NOVELTY_VERSION = 1


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
    ):
        self.panel, self.references, self.min_names = panel, references, min_names
        self.fingerprint = hashlib.sha256(
            json.dumps(references, sort_keys=True).encode()
        ).hexdigest()
        self.sample = np.unique(
            np.linspace(0, len(panel.dates) - 1, min(128, len(panel.dates)), dtype=int)
        )
        self.directory = tempfile.TemporaryDirectory(prefix="alphalineage-novelty-")
        self.memory_budget = max(1, (memory_budget_bytes or 256 * 1024 * 1024) // 4)
        self.items: list[dict[str, Any]] = []
        self.families: list[dict[str, Any]] = []
        self.unavailable: list[str] = []
        self.blocks = []
        self.block_size = 16
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

    def _add(
        self,
        spec: dict[str, Any],
        *,
        known: bool,
        values: np.ndarray | None = None,
        already_ranked: bool = False,
        screened: tuple | None = None,
    ):
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
        path = Path(self.directory.name) / f"{len(self.items)}.npy"
        # Only screen dates need ranks until a reference clears the screening threshold.
        # Preserve raw float64 values on disk so lazy ranking cannot introduce false ties.
        np.save(path, values.astype(np.float32) if already_ranked else values, allow_pickle=False)
        sample = (
            screened[0]
            if screened is not None
            else values[self.sample]
            if already_ranked
            else ranks(values[self.sample])
        ).astype(np.float32)
        # Samples stay mapped as well when the configured memory share is exhausted.
        if sum(item["sample"].nbytes for item in self.items) + sample.nbytes > self.memory_budget:
            sample_path = path.with_suffix(".sample.npy")
            np.save(sample_path, sample, allow_pickle=False)
            sample = np.load(sample_path, mmap_mode="r")
        unit, valid, whole_rows = (
            screened[1:] if screened is not None else standardized(sample, self.min_names)
        )
        index = len(self.items)
        block_index, slot = divmod(index, self.block_size)
        if block_index == len(self.blocks):
            shape = (self.block_size, sample.size)
            if (block_index + 1) * np.prod(shape) * 4 < self.memory_budget // 2:
                block = np.zeros(shape, dtype=np.float32)
            else:
                block = np.lib.format.open_memmap(
                    Path(self.directory.name) / f"block-{block_index}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
            self.blocks.append(block)
        self.blocks[block_index][slot] = unit.ravel()
        self.items.append(
            {
                "ranked": already_ranked,
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

    def _full_ranks(self, item):
        if item["ranked"]:
            return np.load(item["path"], mmap_mode="r")
        raw = np.load(item["path"], mmap_mode="r")
        path = item["path"].with_suffix(".ranks.npy")
        ranked = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=raw.shape)
        for offset in range(0, len(raw), 128):
            ranked[offset : offset + 128] = ranks(raw[offset : offset + 128])
        ranked.flush()
        item["path"], item["ranked"] = path, True
        return ranked

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
        del self.items[mark[0] :]
        del self.families[mark[1] :]

    def apply(self, tree: Node, values: np.ndarray, scored):
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
                other = self._full_ranks(self.items[i])
                if candidate is None:
                    candidate = ranks(values)
                # Stream the full confirmation in fixed date blocks to bound temporary arrays.
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
            # Family prototypes consume disk, not a population's worth of RAM.
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
