"""Deterministic end-to-end multicore training benchmark and release gate.

The default matrix exercises the two shipped search sizes (UI 80x12 and backend
200x25) on 100-, 200-, and 500-symbol panels.  Each worker mode runs in a fresh
subprocess so its elapsed time and peak working-set delta are comparable.  The
canonical population, history, lineage, metrics, and trial count are hashed with
bit-exact float representations; a worker-count mismatch fails the run.

Examples::

    # Fast smoke covering the complete matrix with reduced search sizes.
    python scripts/bench_training.py --quick

    # Release gate (three repeats of the real sizes, including the >=2x Auto gate).
    python scripts/bench_training.py --enforce --output build/training-benchmark.json

    # Focused 200-symbol gate while developing the scorer.
    python scripts/bench_training.py --panels 200 --workloads ui --repeats 3 --enforce
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphalineage.api.resources import (  # noqa: E402
    ResourcePolicy,
    memory_budget_bytes,
    resolve_resources,
)
from alphalineage.core import cpp  # noqa: E402
from alphalineage.core.fitness import forward_returns  # noqa: E402
from alphalineage.core.gp import GP, GPConfig  # noqa: E402
from alphalineage.core.panel import Panel  # noqa: E402
from alphalineage.core.tree import Node, to_json  # noqa: E402

MIB = 1024**2
DEFAULT_PANELS = (100, 200, 500)
DEFAULT_WORKLOADS = ("ui", "backend")
DEFAULT_RESOURCE_MODES = ("serial", "auto", "maximum")
WORKLOADS = {
    "ui": (80, 12),
    "backend": (200, 25),
}
QUICK_WORKLOADS = {
    "ui": (12, 1),
    "backend": (16, 1),
}


@dataclass(frozen=True)
class _Case:
    workload: str
    target_population: int
    target_generations: int
    population: int
    generations: int
    symbols: int
    days: int
    seed: int
    resource_mode: str
    repeat: int
    memory_sample_ms: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class _LineageRecorder:
    """Minimal recorder retaining exactly what worker-count parity must preserve."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def on_init(
        self,
        trees: Sequence[Node],
        *,
        fitnesses: Sequence[float] | None = None,
        ops: Sequence[str] | None = None,
    ) -> None:
        self.events.append(
            {
                "generation": 0,
                "trees": [to_json(tree) for tree in trees],
                "fitnesses": [_exact_float(value) for value in (fitnesses or ())],
                "ops": list(ops or ()),
            }
        )

    def on_generation(
        self,
        generation: int,
        entries: Sequence[tuple[Node, list[int], str, float]],
    ) -> None:
        self.events.append(
            {
                "generation": int(generation),
                "entries": [
                    {
                        "tree": to_json(tree),
                        "parents": list(parents),
                        "op": op,
                        "fitness": _exact_float(fitness),
                    }
                    for tree, parents, op, fitness in entries
                ],
            }
        )


def _exact_float(value: float | int) -> str:
    """Canonical IEEE-754 representation (including signed zero, NaN, and infinities)."""
    return float(value).hex()


def _canonical_hash_value(value: Any) -> Any:
    """Recursively make mixed diagnostics deterministic and JSON-safe for hashing."""
    if isinstance(value, np.generic):
        return _canonical_hash_value(value.item())
    if isinstance(value, float):
        return {"__float_hex__": _exact_float(value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("trajectory mappings must use string keys")
        return {
            key: _canonical_hash_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_hash_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical_hash_value(value.tolist())
    raise TypeError(
        f"unsupported trajectory value {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def _hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical_hash_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory(gp: GP, recorder: _LineageRecorder) -> dict[str, Any]:
    population = [
        {
            "tree": to_json(individual.tree),
            "fitness": _exact_float(individual.fitness),
            "metrics": _canonical_hash_value(individual.metrics),
        }
        for individual in gp.population
    ]
    history = [_canonical_hash_value(row) for row in gp.history]
    components = {
        "population": population,
        "history": history,
        "lineage": recorder.events,
        "metadata": {
            "generation": gp.generation,
            "trial_count": gp.trial_count,
            "termination_reason": gp.termination_reason,
        },
    }
    return {
        "hash": _hash(components),
        "component_hashes": {name: _hash(value) for name, value in components.items()},
        "best_tree": to_json(gp.best(simplified=False).tree),
    }


def _panel(days: int, symbols: int, seed: int) -> Panel:
    """Deterministic, non-degenerate OHLCV panel with a weak learnable signal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="B")
    names = [f"S{index:04d}" for index in range(symbols)]
    latent = rng.normal(0.0, 1.0, (days, symbols))
    cross_signal = latent - latent.mean(axis=1, keepdims=True)
    scale = cross_signal.std(axis=1, keepdims=True)
    cross_signal = np.divide(
        cross_signal,
        scale,
        out=np.zeros_like(cross_signal),
        where=scale != 0.0,
    )
    innovations = 0.002 * np.roll(cross_signal, 1, axis=0)
    innovations[0] = 0.0
    innovations += rng.normal(0.0002, 0.012, (days, symbols))
    close_values = 100.0 * np.cumprod(1.0 + innovations, axis=0)
    close = pd.DataFrame(close_values, index=dates, columns=names)
    open_ = close.shift(1).fillna(close.iloc[0])
    spread = rng.uniform(0.0, 0.006, close.shape)
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close_values) * (1.0 + spread),
        index=dates,
        columns=names,
    )
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close_values) * (1.0 - spread),
        index=dates,
        columns=names,
    )
    volume = pd.DataFrame(
        1_000_000.0 * np.exp(0.25 * latent),
        index=dates,
        columns=names,
    )
    return Panel.from_prices(open=open_, high=high, low=low, close=close, volume=volume)


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _rss_bytes() -> tuple[int, str]:
    """Current process RSS without introducing a benchmark-only runtime dependency."""
    if os.name == "nt":
        counters = _WindowsProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return int(counters.working_set_size), "windows-working-set"
    proc_statm = Path("/proc/self/statm")
    if proc_statm.exists():
        pages = int(proc_statm.read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE")), "proc-rss"
    try:
        import resource

        high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        multiplier = 1 if sys.platform == "darwin" else 1024
        return high_water * multiplier, "process-high-water"
    except (ImportError, OSError, ValueError):
        return 0, "unavailable"


class _MemoryMonitor:
    def __init__(self, sample_ms: int) -> None:
        self.sample_seconds = max(0.001, sample_ms / 1000.0)
        self.baseline_bytes, self.source = _rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="benchmark-rss", daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.sample_seconds):
            rss, _source = _rss_bytes()
            self.peak_bytes = max(self.peak_bytes, rss)

    def __enter__(self) -> _MemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        rss, _source = _rss_bytes()
        self.peak_bytes = max(self.peak_bytes, rss)
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)


def _resource_settings(mode: str) -> tuple[int, int, dict[str, Any]]:
    accelerated = cpp.backend_enabled()
    if mode == "serial":
        guard = memory_budget_bytes()
        return (
            1,
            guard,
            {
                "profile": "serial",
                "percent": 0,
                "workers": 1,
                "memory_budget_bytes": guard,
                "run_memory_budget_bytes": guard,
                "accelerated": accelerated,
                "fallback_reason": cpp.unavailable_reason() if not accelerated else None,
            },
        )
    resolved = resolve_resources(
        ResourcePolicy(mode),  # type: ignore[arg-type]
        accelerated=accelerated,
        fallback_reason=cpp.unavailable_reason(),
    )
    return resolved.workers, resolved.run_memory_budget_bytes, resolved.to_dict()


def _run_child(case: _Case) -> dict[str, Any]:
    panel = _panel(case.days, case.symbols, case.seed)
    fwd = forward_returns(panel, 1)
    workers, guard, resources = _resource_settings(case.resource_mode)
    recorder = _LineageRecorder()
    config = GPConfig(
        population_size=case.population,
        generations=case.generations,
        seed=case.seed,
    )
    gp = GP(
        config,
        panel,
        fwd,
        recorder=recorder,
        workers=workers,
        memory_budget_bytes=guard,
    )

    # Exclude one-time extension import and panel-array materialization from each timed run.
    if cpp.backend_enabled():
        cpp.evaluate_cpp(Node("close"), panel)
    gc.collect()
    with _MemoryMonitor(case.memory_sample_ms) as memory:
        start = time.perf_counter()
        gp.run()
        elapsed = time.perf_counter() - start

    trajectory = _trajectory(gp, recorder)
    result = {
        **case.to_dict(),
        "workers": workers,
        "native_available": cpp.available(),
        "native_enabled": cpp.backend_enabled(),
        "elapsed_seconds": elapsed,
        "trials": gp.trial_count,
        "factors_per_second": gp.trial_count / elapsed if elapsed > 0.0 else None,
        "rss_source": memory.source,
        "baseline_rss_bytes": memory.baseline_bytes,
        "peak_rss_bytes": memory.peak_bytes,
        "peak_rss_delta_bytes": memory.delta_bytes,
        "memory_guard_bytes": guard,
        "memory_guard_ok": memory.source != "unavailable" and memory.delta_bytes <= guard,
        "resources": resources,
        "trajectory": trajectory,
    }
    return result


def _parse_csv(raw: str, *, choices: set[str] | None = None) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty")
    if choices is not None and (invalid := sorted(set(values) - choices)):
        raise argparse.ArgumentTypeError(f"unknown value(s): {', '.join(invalid)}")
    return values


def _parse_panels(raw: str) -> list[int]:
    try:
        panels = [int(value) for value in _parse_csv(raw)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("panels must be comma-separated integers") from exc
    if any(value < 2 for value in panels):
        raise argparse.ArgumentTypeError("panel sizes must be at least 2")
    return panels


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="cover the selected matrix with 12x1/16x1 searches and one repeat",
    )
    parser.add_argument(
        "--panels",
        type=_parse_panels,
        default=list(DEFAULT_PANELS),
        help="comma-separated symbol counts (default: 100,200,500)",
    )
    parser.add_argument(
        "--workloads",
        type=lambda value: _parse_csv(value, choices=set(WORKLOADS)),
        default=list(DEFAULT_WORKLOADS),
        help="comma-separated ui,backend (default: both)",
    )
    parser.add_argument(
        "--resources",
        type=lambda value: _parse_csv(value, choices=set(DEFAULT_RESOURCE_MODES)),
        default=list(DEFAULT_RESOURCE_MODES),
        help="comma-separated serial,auto,maximum (default: all)",
    )
    parser.add_argument("--days", type=int, help="history rows (default: 504, or 128 quick)")
    parser.add_argument("--repeats", type=int, help="timing repeats (default: 3, or 1 quick)")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--memory-sample-ms", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="fail unless full 200-symbol Auto medians meet --min-speedup",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument("--_case-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_result-file", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.days is not None and args.days < 30:
        parser.error("--days must be at least 30")
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.memory_sample_ms < 1:
        parser.error("--memory-sample-ms must be positive")
    if args.min_speedup <= 0.0:
        parser.error("--min-speedup must be positive")
    if args.enforce and args.quick:
        parser.error("--enforce requires the real workload sizes; remove --quick")
    if args.enforce and (
        200 not in args.panels or not {"serial", "auto"}.issubset(args.resources)
    ):
        parser.error("--enforce requires panel 200 and both serial and auto resources")
    if bool(args._case_file) != bool(args._result_file):
        parser.error("internal case and result paths must be supplied together")


def _child_entry(case_path: Path, result_path: Path) -> int:
    case = _Case(**json.loads(case_path.read_text(encoding="utf-8")))
    result = _run_child(case)
    result_path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    return 0


def _execute_case(case: _Case, temp_dir: Path, ordinal: int) -> dict[str, Any]:
    case_path = temp_dir / f"case-{ordinal}.json"
    result_path = temp_dir / f"result-{ordinal}.json"
    case_path.write_text(json.dumps(case.to_dict()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_case-file",
            str(case_path),
            "--_result-file",
            str(result_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not result_path.exists():
        details = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise RuntimeError(
            f"benchmark child failed for {case.workload}/{case.symbols}/"
            f"{case.resource_mode}: {details}"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def _case_key(result: dict[str, Any]) -> tuple[str, int]:
    return str(result["workload"]), int(result["symbols"])


def _median(results: Sequence[dict[str, Any]], field: str) -> float:
    return statistics.median(float(result[field]) for result in results)


def _analyse(
    results: list[dict[str, Any]], *, quick: bool, minimum_speedup: float
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(_case_key(result), []).append(result)

    parity_failures: list[dict[str, Any]] = []
    memory_failures: list[dict[str, Any]] = []
    speedups: dict[str, dict[str, float]] = {}
    speed_gate_cases: list[dict[str, Any]] = []
    speed_gate_failures: list[dict[str, Any]] = []
    for key, case_results in grouped.items():
        hashes = {result["trajectory"]["hash"] for result in case_results}
        if len(hashes) != 1:
            parity_failures.append(
                {
                    "workload": key[0],
                    "symbols": key[1],
                    "hashes": {
                        f"{result['resource_mode']}#{result['repeat']}": result["trajectory"][
                            "component_hashes"
                        ]
                        for result in case_results
                    },
                }
            )
        memory_failures.extend(
            {
                "workload": result["workload"],
                "symbols": result["symbols"],
                "resource_mode": result["resource_mode"],
                "repeat": result["repeat"],
                "peak_rss_delta_bytes": result["peak_rss_delta_bytes"],
                "memory_guard_bytes": result["memory_guard_bytes"],
                "rss_source": result["rss_source"],
            }
            for result in case_results
            if not result["memory_guard_ok"]
        )
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for result in case_results:
            by_mode.setdefault(str(result["resource_mode"]), []).append(result)
        if "serial" not in by_mode:
            continue
        serial_seconds = _median(by_mode["serial"], "elapsed_seconds")
        key_name = f"{key[0]}-{key[1]}"
        speedups[key_name] = {}
        for mode in ("auto", "maximum"):
            if mode not in by_mode:
                continue
            mode_seconds = _median(by_mode[mode], "elapsed_seconds")
            speedup = serial_seconds / mode_seconds
            speedups[key_name][mode] = speedup
            if not quick and key[1] == 200 and mode == "auto":
                gate_case = {
                    "workload": key[0],
                    "symbols": key[1],
                    "serial_seconds": serial_seconds,
                    "auto_seconds": mode_seconds,
                    "speedup": speedup,
                    "required": minimum_speedup,
                }
                speed_gate_cases.append(gate_case)
                if speedup < minimum_speedup:
                    speed_gate_failures.append(gate_case)

    native_ready = all(bool(result["native_enabled"]) for result in results)
    return {
        "native_ready": native_ready,
        "parity_ok": not parity_failures,
        "memory_ok": not memory_failures,
        "speed_gate_applicable": bool(speed_gate_cases),
        "speed_gate_ok": not speed_gate_failures,
        "speedups": speedups,
        "speed_gate_cases": speed_gate_cases,
        "parity_failures": parity_failures,
        "memory_failures": memory_failures,
        "speed_gate_failures": speed_gate_failures,
    }


def _human_bytes(value: float) -> str:
    return f"{value / MIB:.1f} MiB"


def _print_results(results: list[dict[str, Any]], gates: dict[str, Any]) -> None:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (str(result["workload"]), int(result["symbols"]), str(result["resource_mode"]))
        grouped.setdefault(key, []).append(result)

    print(
        "workload  symbols  mode       workers   median s   factors/s   peak delta / guard"
    )
    print("-" * 88)
    for (workload, symbols, mode), rows in grouped.items():
        elapsed = _median(rows, "elapsed_seconds")
        rate = _median(rows, "factors_per_second")
        peak = max(int(row["peak_rss_delta_bytes"]) for row in rows)
        guard = min(int(row["memory_guard_bytes"]) for row in rows)
        workers = sorted({int(row["workers"]) for row in rows})
        worker_label = str(workers[0]) if len(workers) == 1 else str(workers)
        print(
            f"{workload:<9} {symbols:>7}  {mode:<10} {worker_label:>7}  "
            f"{elapsed:>9.3f}  {rate:>10.1f}   {_human_bytes(peak):>10} / "
            f"{_human_bytes(guard)}"
        )

    print()
    print(f"exact trajectory parity: {'PASS' if gates['parity_ok'] else 'FAIL'}")
    print(f"peak-memory guard:       {'PASS' if gates['memory_ok'] else 'FAIL'}")
    print(f"native evaluator:        {'PASS' if gates['native_ready'] else 'FAIL'}")
    if gates["speed_gate_applicable"]:
        print(f"200-symbol Auto speed:   {'PASS' if gates['speed_gate_ok'] else 'FAIL'}")
    else:
        print("200-symbol Auto speed:   SKIP (no full-size 200-symbol Auto comparison)")
    for case, values in gates["speedups"].items():
        rendered = ", ".join(f"{mode}={speedup:.2f}x" for mode, speedup in values.items())
        print(f"  {case}: {rendered}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args._case_file is not None:
        return _child_entry(args._case_file, args._result_file)

    days = args.days if args.days is not None else (128 if args.quick else 504)
    repeats = args.repeats if args.repeats is not None else (1 if args.quick else 3)
    temp_dir = ROOT / ".tmp" / f"bench-training-{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    cases: list[_Case] = []
    for workload in args.workloads:
        target_population, target_generations = WORKLOADS[workload]
        population, generations = (
            QUICK_WORKLOADS[workload]
            if args.quick
            else (target_population, target_generations)
        )
        for symbols in args.panels:
            for resource_mode in args.resources:
                for repeat in range(1, repeats + 1):
                    cases.append(
                        _Case(
                            workload=workload,
                            target_population=target_population,
                            target_generations=target_generations,
                            population=population,
                            generations=generations,
                            symbols=symbols,
                            days=days,
                            seed=args.seed,
                            resource_mode=resource_mode,
                            repeat=repeat,
                            memory_sample_ms=args.memory_sample_ms,
                        )
                    )

    results: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{len(cases)}] {case.workload} "
                f"{case.population}x{case.generations}, {case.symbols} symbols, "
                f"{case.resource_mode}, repeat {case.repeat}",
                flush=True,
            )
            results.append(_execute_case(case, temp_dir, index))
    finally:
        for path in temp_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        temp_dir.rmdir()

    gates = _analyse(results, quick=args.quick, minimum_speedup=args.min_speedup)
    _print_results(results, gates)
    payload = {
        "schema_version": 1,
        "quick": bool(args.quick),
        "days": days,
        "repeats": repeats,
        "minimum_auto_speedup": args.min_speedup,
        "results": results,
        "gates": gates,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"JSON: {args.output.resolve()}")

    mandatory_ok = gates["parity_ok"] and gates["memory_ok"]
    if args.enforce:
        mandatory_ok = mandatory_ok and gates["native_ready"] and gates["speed_gate_ok"]
    return 0 if mandatory_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
