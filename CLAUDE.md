# CLAUDE.md

## What this is
Evolutionary (genetic-programming) alpha-factor mining platform. Strategies are
strongly-typed expression trees, evolved via GP, scored by IC, validated with a
built-in anti-overfitting suite.

## Invariants (never violate)
1. The test split is locked. Any user-facing metric defaults to OOS/deflated.
2. Encode strategies as typed expression trees, never flat numeric vectors.
3. GP uses a population + crossover, not single-point hill climbing.
4. Fitness is IC / rank IC, not raw PnL.
5. User operators go through a typed, vectorized DSL. No arbitrary server-side code.
6. Backtests include transaction costs; signals must survive them.
7. Universe is point-in-time; corporate actions adjusted; no survivorship bias.
8. Not investment advice, not a brokerage. Disclaimers on outputs.

## Commands
- Setup: `uv venv && uv pip install -e ".[dev]"`
- Test:  `pytest -q --cov=alphalineage`
- Lint:  `ruff check . && ruff format --check .`
- Types: `mypy src`
- Data:  `python scripts/download_universe.py --universe sp500-lite --years 15`
- Run (CLI): `python scripts/run_gp.py --config configs/dev.yaml`
- Run (app): `docker compose up` → http://localhost:8000 (single image; the API serves the built UI).
  Fallback: `uvicorn alphalineage.api.app:app --port 8000` + `cd frontend && npm run dev:app`.
- Demo export: `python scripts/export_demo.py --workspace run-<id> --out frontend/public/demo-run.json`

## Iterative sessions (V1 finish)
- A *session* (`src/alphalineage/api/sessions.py`) is a search grown over *segments*: create →
  continue (warm-start the evolved population) with changed config/universe/operators → seed new
  sessions from saved factors (`src/alphalineage/library/factors.py`). State lives under
  `data_cache/sessions/{id}/` (session.json + checkpoint.json + lineage.json + result.json).
- Honesty across segments: the train/test **time boundary** is frozen at creation (never relocated,
  even when the universe changes); trial counts are cumulative + monotone (carried through the GP
  checkpoint); every segment is one OOS read, counted and surfaced in the dashboard.
- Factor storage dir is user-configurable: env `ALPHALINEAGE_FACTORS_DIR` > `meta/settings.json`
  (`PUT /settings`) > default. Lineage nodes carry `fitness`, powering the grouped genealogy view.

## Conventions
- Python 3.11+. One task = one PR-sized change. Write the acceptance test first.
- Hot paths (tree evaluation) must be vectorized (numpy/pandas/numba).
- The evaluator has an **optional C++ backend** (`cpp/`, pybind11+CMake; build via
  `python scripts/build_cpp.py`). Pure Python is the default + correctness baseline and always runs
  without a compiler; the C++ backend auto-engages when built (`ALPHALINEAGE_EVALUATOR=auto|python|cpp`)
  and is pinned identical to Python by the parity test. The compiled `.pyd`/`.so` is gitignored.
- Do not call the data API inside the GP loop; read from the Parquet cache.
- The two load-bearing tests are synthetic-signal recovery and noise rejection.
  If either regresses, stop and fix before anything else.

## Backend & distribution (decided Phase 5)
- Backend stack is **lightweight, no external servers**: FastAPI + an in-process threaded job
  runner + JSON/SQLite. RQ/Redis + Postgres are only a future production swap behind the
  job/store interfaces — do not introduce them into the dev stack.
- Two frontend build targets: **`demo`** (static, JSON-driven snapshot of one finished run —
  tree + genealogy + OOS/deflated metrics — no backend, deploys to a static host) and **`app`**
  (talks to the local FastAPI+jobs+SQLite backend; real searches with the user's Tiingo key).
  They are distinct builds (CORS) and not connected by default.
- Local run is **Docker-first** (`docker compose up`; uv/pip is the fallback). Docker is a
  packaging convenience only — it changes neither the architecture nor any invariant, and is the
  final pre-ship step (Phase P), not started until the Phase 5/6/7 gates are green.

## Phase order
Phases 0 → 8 (plus Phase P, Packaging & Delivery, near the end of V1) in DEVELOPMENT_PLAN.md.
Do not start a phase until the prior gate is green.