# AlphaForge

Evolutionary (genetic-programming) alpha-factor mining platform. Strategies are
strongly-typed expression trees, evolved via GP, scored by IC, and validated with a
built-in anti-overfitting suite.

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the phased roadmap and
[CLAUDE.md](CLAUDE.md) for the standing invariants.

## Status

**Phase 0 — Data foundation.** Cached, corporate-action-adjusted, point-in-time
universe data behind a provider abstraction (Tiingo, with a yfinance fallback).

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env   # then paste your TIINGO_API_KEY
```

## Quality gates

```bash
pytest -q --cov=alphaforge
ruff check . && ruff format --check .
mypy src
```

## Pull data

```bash
python scripts/download_universe.py --universe sp500-lite --years 15
```

Data is cached locally under `data_cache/` (gitignored). The data API is fetched once
and cached — it is never called inside the GP loop.

## Disclaimer

This software is for research and educational use only. It is **not investment advice**
and **not a brokerage**. Nothing here is a recommendation to buy or sell any security,
and there is no promise of beating the market.
