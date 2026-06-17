# AlphaLineage

Evolutionary genetic-programming alpha-factor mining platform. Strategies are strongly typed
expression trees, evolved via genetic programming, scored by information coefficient (IC), and
validated with a built-in anti-overfitting suite. You drive the whole thing from a browser —
define a universe, add custom operators, train, watch it evolve, save the factors you like, and
seed new searches from them — without ever touching the backend directly.

The Python package is `alphalineage`. **New here?** Read [docs/TUTORIAL.md](docs/TUTORIAL.md) — it
explains every component in plain language and walks through the UI step by step.

## Status

V1: cached point-in-time data, typed factor trees, GP search, anti-overfitting validation,
backtests with costs, a FastAPI + in-process-jobs backend, an optional C++ evaluator, and a React
UI for training, iterating, the factor library, and an honest metrics/genealogy view.

---

## Run it (recipe)

### 1. Docker — the one-command path (recommended)

You need only [Docker](https://docs.docker.com/get-docker/) installed.

```bash
git clone <this-repo> && cd alphalineage
cp .env.example .env          # optional: paste a free Tiingo key to download fresh data
docker compose up             # builds the UI + backend into one image and serves it
```

Open **http://localhost:8000**. That is the whole app — the FastAPI backend serves the built UI
on the same origin. The bundled `data_cache/` already contains a small `sp500-lite` universe, so
you can train immediately without a data key. Your saved factors, sessions, and any downloaded
data persist on the host under `./data_cache/`.

To stop: `Ctrl-C`, then `docker compose down`.

### 2. No Docker — Python + Node fallback

Requires Python 3.11+ and Node 20+. Any virtualenv tool works (`uv`, `venv`, or conda):

```bash
# backend
python -m venv .venv && . .venv/Scripts/activate   # (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
cp .env.example .env                                # optional Tiingo key
uvicorn alphalineage.api.app:app --port 8000        # API on :8000
```

```bash
# frontend (second terminal)
cd frontend
npm install
npm run dev:app                                     # UI on :5173, talks to the backend on :8000
```

Open **http://localhost:5173**.

> `uv venv && uv pip install -e ".[dev]"` is the fastest install when `uv` is available.

### Zero-backend demo

A static snapshot of one finished run (metrics, factor tree, genealogy) — no backend, deploys to
any static host:

```bash
cd frontend && npm run build:demo      # outputs dist/, reads public/demo-run.json
npm run preview                        # or serve dist/ anywhere
```

Regenerate the demo snapshot from a real run with
`python scripts/export_demo.py --workspace run-<id> --out frontend/public/demo-run.json`.

---

## First run, in the browser

1. **Train** tab → pick a universe (`sp500-lite` is bundled), set the GP knobs (or keep the
   defaults), **Start training**. Watch the generation bar and fitness sparkline.
2. **Metrics / Best factor / Genealogy** → inspect the result. Metrics default to out-of-sample,
   deflated values (the honest numbers); the genealogy groups each generation by how individuals
   were bred and lets you trace any factor's ancestry.
3. **Continue** from the finished run with more generations or changed settings — or **save** a
   factor to the Library and **seed** a brand-new session from it.
4. **Extend** tab → define a custom point-in-time universe or compose a new typed operator.

See [docs/TUTORIAL.md](docs/TUTORIAL.md) for the full walkthrough.

## Quality gates

```bash
pytest -q                       # backend tests (synthetic-signal recovery + noise rejection are load-bearing)
ruff check . && ruff format --check .
mypy src
cd frontend && npm run typecheck && npm test
```

## Pull more data

```bash
python scripts/download_universe.py --universe sp500-lite --years 15
```

Data is cached under `data_cache/` (gitignored), fetched once and never called inside the GP loop.
A free [Tiingo](https://www.tiingo.com) key in `.env` enables downloads; yfinance is the fallback.

## Disclaimer

This software is for research and educational use only. It is **not investment advice** and not a
brokerage. Nothing here is a recommendation to buy or sell any security, and there is no promise
of beating the market.
