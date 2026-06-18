import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const addFormula = vi.fn();
const defineUniverse = vi.fn();
const deleteFormula = vi.fn();
const deleteUniverse = vi.fn();
const getDataCoverage = vi.fn();
const getDataSync = vi.fn();
const getUniverse = vi.fn();
const listFormulas = vi.fn();
const listUniverses = vi.fn();
const searchSymbols = vi.fn();
const startDataSync = vi.fn();
const updateUniverse = vi.fn();
const validateSymbol = vi.fn();

vi.mock("../api/client", () => ({
  addFormula: (payload: unknown) => addFormula(payload),
  defineUniverse: (payload: unknown) => defineUniverse(payload),
  deleteFormula: (name: string) => deleteFormula(name),
  deleteUniverse: (name: string) => deleteUniverse(name),
  getDataCoverage: (...args: unknown[]) => getDataCoverage(...args),
  getDataSync: (jobId: string) => getDataSync(jobId),
  getUniverse: (name: string) => getUniverse(name),
  listFormulas: () => listFormulas(),
  listUniverses: () => listUniverses(),
  searchSymbols: (query: string) => searchSymbols(query),
  startDataSync: (payload: unknown) => startDataSync(payload),
  updateUniverse: (name: string, payload: unknown) => updateUniverse(name, payload),
  validateSymbol: (payload: unknown) => validateSymbol(payload),
}));

import { UniverseUpdater } from "./UniverseUpdater";

function setupMocks() {
  listUniverses.mockResolvedValue([
    {
      name: "sp500-lite",
      symbols: ["AAPL"],
      source: "sample",
      memberships: [{ symbol: "AAPL", entry: "2020-01-01", exit: null }],
    },
  ]);
  listFormulas.mockResolvedValue([]);
  getUniverse.mockResolvedValue({
    name: "sp500-lite",
    symbols: ["AAPL"],
    source: "sample",
    memberships: [{ symbol: "AAPL", entry: "2020-01-01", exit: null }],
  });
  searchSymbols.mockResolvedValue([
    {
      symbol: "AAPL",
      name: "Apple Inc.",
      exchange: "Nasdaq",
      quote_type: "Equity",
      currency: "USD",
      source: "yfinance",
    },
  ]);
  validateSymbol.mockResolvedValue({
    symbol: "AAPL",
    valid: true,
    rows: 100,
    first_date: "2020-01-02",
    last_date: "2026-06-16",
    provider: "yfinance",
    error: null,
  });
  getDataCoverage.mockResolvedValue([
    {
      symbol: "AAPL",
      cached: true,
      rows: 20,
      first_date: "2020-01-02",
      last_date: "2020-02-01",
      requested_start: "2020-01-01",
      requested_end: "2026-06-17",
      needs_sync: true,
    },
  ]);
  startDataSync.mockResolvedValue({ job_id: "sync-1", status: "queued" });
  getDataSync.mockResolvedValue({
    job_id: "sync-1",
    status: "done",
    result: {
      mode: "refresh",
      start: "2020-01-01",
      end: null,
      results: [
        {
          symbol: "AAPL",
          status: "fetched",
          rows_fetched: 100,
          rows_cached: 100,
          first_date: "2020-01-02",
          last_date: "2026-06-16",
          provider: "yfinance",
          error: null,
        },
      ],
    },
    error: null,
  });
  addFormula.mockImplementation(async (payload) => ({ ...payload, registered: true, error: null }));
  defineUniverse.mockResolvedValue({ name: "my-universe", symbols: ["AAPL"] });
  updateUniverse.mockResolvedValue({ name: "my-universe", symbols: ["AAPL"] });
  deleteUniverse.mockResolvedValue(undefined);
  deleteFormula.mockResolvedValue(undefined);
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("UniverseUpdater", () => {
  it("searches ticker candidates, validates one, and adds it to the draft", async () => {
    setupMocks();
    render(<UniverseUpdater />);

    fireEvent.click(screen.getByRole("button", { name: /Add ticker/ }));
    fireEvent.change(screen.getByLabelText("Ticker query"), { target: { value: "AAPL" } });
    fireEvent.click(screen.getByTestId("search-symbols"));

    const candidates = await screen.findByTestId("symbol-candidates");
    fireEvent.click(within(candidates).getByText("AAPL"));
    expect(await screen.findByTestId("symbol-validation")).toHaveTextContent("validated");

    fireEvent.click(screen.getByText("Add selected ticker"));
    expect(screen.getByLabelText("symbol-0")).toHaveValue("AAPL");
  });

  it("keeps sync controls collapsed by default and can choose refresh mode", async () => {
    setupMocks();
    render(
      <UniverseUpdater
        draft={{
          name: "my-universe",
          rows: [{ symbol: "AAPL", entry: "2020-01-01", exit: "" }],
        }}
      />,
    );

    expect(screen.queryByLabelText("Sync start date")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Sync data/ }));

    const incremental = screen.getByLabelText("Incremental merge");
    expect(incremental).toBeChecked();
    fireEvent.click(incremental);
    expect(incremental).not.toBeChecked();

    fireEvent.click(screen.getByTestId("sync-universe"));
    await waitFor(() =>
      expect(startDataSync).toHaveBeenCalledWith(
        expect.objectContaining({ mode: "refresh", symbols: ["AAPL"] }),
      ),
    );
  });

  it("saves a formula template through the formula API", async () => {
    setupMocks();
    render(<UniverseUpdater />);

    fireEvent.click(screen.getByRole("button", { name: /Formula library/ }));
    fireEvent.click(screen.getByText("Macd-style spread"));
    fireEvent.change(screen.getByLabelText("Formula name"), { target: { value: "macd_signal" } });
    fireEvent.click(screen.getByTestId("save-formula"));

    await waitFor(() =>
      expect(addFormula).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "macd_signal",
          arg_types: ["series"],
          out_type: "series",
          body: expect.objectContaining({ name: "sub" }),
        }),
      ),
    );
  });
});
