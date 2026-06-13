# AlphaLineage

Evolutionary genetic-programming alpha-factor mining platform. Strategies are
strongly typed expression trees, evolved via GP, scored by IC, and validated with a
built-in anti-overfitting suite.

The Python package is `alphalineage`.

## Status

Phases 0-7 are implemented as the local research app: cached point-in-time data,
typed factor trees, GP search, anti-overfitting validation, backtests, FastAPI jobs,
React visualization, custom universes, custom operators, and workspace persistence.

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env   # then paste your TIINGO_API_KEY
```

## Open Locally

Use the helper scripts from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-AlphaLineage.ps1 -Mode demo
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-AlphaLineage.ps1 -Mode app
```

Both modes open at `http://localhost:5173`. App mode also starts the FastAPI backend
at `http://localhost:8000`.

The scripts write PID files and logs under `.runtime/`. Add `-Restart` to the start
command to stop any script-started servers before launching again.

Stop the local servers:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Stop-AlphaLineage.ps1
```

Manual static demo, no backend:

```bash
cd frontend
npm run dev -- --mode demo
```

Open `http://localhost:5173`.

Manual backend-enabled app, two terminals:

```powershell
$env:TEMP='F:\Alpha-Factor Mining\.tmp'
$env:TMP='F:\Alpha-Factor Mining\.tmp'
$env:PYTHONPATH='F:\Alpha-Factor Mining\src'
.venv\Scripts\python.exe -m uvicorn alphalineage.api.app:app --reload --port 8000
```

```bash
cd frontend
npm run dev:app
```

Open `http://localhost:5173`.

## Quality Gates

```bash
$env:TEMP='F:\Alpha-Factor Mining\.tmp'; $env:TMP='F:\Alpha-Factor Mining\.tmp'
.venv\Scripts\python.exe -m pytest -q
ruff check . && ruff format --check .
mypy src
cd frontend && npm run typecheck && npm test
```

## Pull Data

```bash
python scripts/download_universe.py --universe sp500-lite --years 15
```

Data is cached locally under `data_cache/` (gitignored). The data API is fetched once
and cached; it is never called inside the GP loop.

## Disclaimer

This software is for research and educational use only. It is not investment advice
and not a brokerage. Nothing here is a recommendation to buy or sell any security,
and there is no promise of beating the market.
