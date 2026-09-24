"""Isolated browser fixture through 2026-09-23; never contacts a market-data provider.

Build into .runtime/workflow-qa-dist with VITE_API_BASE=http://127.0.0.1:8001.
Run this script, select universe workflow-qa and use an as-of date of 2026-09-23.
The first visit has ten cached stocks and two uncached stocks for preparation.
"""

import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ["ALPHALINEAGE_SKIP_DOTENV"] = "1"
    os.environ["ALPHALINEAGE_DATA_DIR"] = str(
        root / ".runtime" / "workflow-verification" / "workflow-qa-data"
    )
    os.environ["ALPHALINEAGE_STATIC_DIR"] = str(root / ".runtime" / "workflow-qa-dist")

    import numpy as np
    import pandas as pd
    import uvicorn

    import alphalineage.api.app as api
    from alphalineage.data.cache import ParquetCache
    from alphalineage.data.universe import Membership, Universe

    symbols = [f"QA{i:02}" for i in range(12)]
    universe = Universe(
        "workflow-qa", [Membership(symbol, pd.Timestamp("2024-01-02")) for symbol in symbols]
    )
    universe.save()

    class SyntheticProvider:
        name = "synthetic-qa"

        def get_prices(self, symbol, start, end):
            if symbol not in symbols:
                raise ValueError("This isolated fixture only provides workflow-qa stocks")
            dates = pd.bdate_range("2024-01-02", "2026-09-23", name="date")
            rng = np.random.default_rng(int(symbol[-2:]) + 31)
            close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, len(dates))))
            result = pd.DataFrame(
                {
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.uniform(1e6, 3e6, len(dates)),
                    "div_cash": 0.0,
                    "split_factor": 1.0,
                },
                index=dates,
            )
            return result.loc[start:end]

    api._price_provider = lambda: SyntheticProvider()
    cache = ParquetCache()
    for symbol in symbols[:10]:
        if not cache.has(symbol):
            cache.store(symbol, SyntheticProvider().get_prices(symbol, "2024-01-02", "2026-09-22"))
    print("Isolated synthetic QA: http://127.0.0.1:8001", flush=True)
    uvicorn.run(api.app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
