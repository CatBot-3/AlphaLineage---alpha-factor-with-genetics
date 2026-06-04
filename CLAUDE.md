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
- Test:  `pytest -q --cov=alphaforge`
- Lint:  `ruff check . && ruff format --check .`
- Types: `mypy src`
- Data:  `python scripts/download_universe.py --universe sp500-lite --years 15`
- Run:   `python scripts/run_gp.py --config configs/dev.yaml`

## Conventions
- Python 3.11+. One task = one PR-sized change. Write the acceptance test first.
- Hot paths (tree evaluation) must be vectorized (numpy/pandas/numba).
- Do not call the data API inside the GP loop; read from the Parquet cache.
- The two load-bearing tests are synthetic-signal recovery and noise rejection.
  If either regresses, stop and fix before anything else.

## Phase order
Phases 0 → 8 in DEVELOPMENT_PLAN.md. Do not start a phase until the prior gate is green.