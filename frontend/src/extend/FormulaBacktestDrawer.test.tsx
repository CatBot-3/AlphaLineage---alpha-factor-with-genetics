import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const listUniverses = vi.fn();
const listFormulaTests = vi.fn();
const listFormulaResults = vi.fn();
const startFormulaTest = vi.fn();
const getFormulaTest = vi.fn();
const keepFormulaTest = vi.fn();

vi.mock("../api/client", () => ({
  listUniverses: () => listUniverses(),
  listFormulaTests: () => listFormulaTests(),
  listFormulaResults: () => listFormulaResults(),
  startFormulaTest: (request: unknown) => startFormulaTest(request),
  getFormulaTest: (id: string) => getFormulaTest(id),
  keepFormulaTest: (id: string, request: unknown) => keepFormulaTest(id, request),
  clearFormulaTest: vi.fn().mockResolvedValue(undefined),
  stopFormulaTest: vi.fn().mockResolvedValue({ stopping: true }),
}));

import { FormulaBacktestDrawer } from "./FormulaBacktestDrawer";

const SOURCE = {
  kind: "draft" as const,
  body: { name: "rank", children: [{ name: "close" }] },
  inputs: [],
  out_type: "signal",
};

function setup() {
  listUniverses.mockResolvedValue([{
    name: "sample",
    display_name: "Sample universe",
    source: "sample",
    symbols: ["AAA", "BBB"],
    memberships: [],
    cache_coverage: {
      as_of: "2025-12-31",
      eligible_symbols: ["AAA", "BBB"],
      cached_symbols: ["AAA", "BBB"],
      missing_symbols: [],
      complete: true,
      symbol_coverage: {
        AAA: { first_date: "2020-01-02", last_date: "2025-12-31", issues: [] },
        BBB: { first_date: "2020-02-03", last_date: "2025-12-30", issues: [] },
      },
    },
  }]);
  listFormulaTests.mockResolvedValue([]);
  listFormulaResults.mockResolvedValue([]);
  startFormulaTest.mockResolvedValue({ job_id: "job-1", status: "queued" });
  getFormulaTest.mockResolvedValue({ job_id: "job-1", status: "queued", result: null, error: null });
}

afterEach(() => vi.clearAllMocks());

describe("FormulaBacktestDrawer", () => {
  it("runs an unsaved draft with visible research defaults", async () => {
    setup();
    render(<FormulaBacktestDrawer open source={SOURCE} inputs={[]} formulas={[]} defaultName="Draft test" onClose={() => undefined} />);

    expect(await screen.findByDisplayValue("Sample universe")).toBeInTheDocument();
    expect(await screen.findByDisplayValue("2020-01-02")).toBeInTheDocument();
    expect(await screen.findByDisplayValue("2025-12-31")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));

    await waitFor(() => expect(startFormulaTest).toHaveBeenCalledWith(expect.objectContaining({
      source: SOURCE,
      bindings: {},
      universe: "sample",
      horizon: 1,
      weighting_scheme: "quantile_ls",
      quantile: 0.2,
      commission_bps: 1,
      slippage_bps: 5,
    })));
  });

  it("requires an explicit binding for every exposed input", async () => {
    setup();
    const inputs = [{ name: "price", type: "series", description: "Series to transform." }];
    render(<FormulaBacktestDrawer open source={{ ...SOURCE, inputs, body: { name: "rank", children: [{ name: "$arg", value: 0 }] } }} inputs={inputs} formulas={[]} defaultName="Bound test" onClose={() => undefined} />);

    await screen.findByDisplayValue("Sample universe");
    expect(screen.getByRole("button", { name: "Run backtest" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/price/), { target: { value: "data:close" } });
    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));

    await waitFor(() => expect(startFormulaTest).toHaveBeenCalledWith(expect.objectContaining({
      bindings: { price: { kind: "field", field: "close" } },
    })));
  });

  it("does not silently apply an authoring default to a numeric test input", async () => {
    setup();
    const inputs = [{ name: "lookback", type: "window", description: "Rolling lookback.", default: 10 }];
    render(<FormulaBacktestDrawer open source={{ ...SOURCE, inputs, body: { name: "ts_mean", children: [{ name: "close" }, { name: "$arg", value: 0 }] } }} inputs={inputs} formulas={[]} defaultName="Window test" onClose={() => undefined} />);

    await screen.findByDisplayValue("Sample universe");
    const field = screen.getByLabelText(/lookback/);
    expect(field).toHaveValue(null);
    expect(screen.getByRole("button", { name: "Run backtest" })).toBeDisabled();
    fireEvent.change(field, { target: { value: "10" } });
    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));

    await waitFor(() => expect(startFormulaTest).toHaveBeenCalledWith(expect.objectContaining({
      bindings: { lookback: { kind: "literal", value: 10 } },
    })));
  });

  it("keeps a completed test only after the user asks", async () => {
    setup();
    listFormulaTests.mockResolvedValue([{
      job_id: "job-done",
      status: "done",
      error: null,
      result: {
        kind: "backtest",
        tree: SOURCE.body,
        source: SOURCE,
        bindings: {},
        dependency_revisions: [],
        expression_fingerprint: "abc",
        universe: "sample",
        start: "2020-01-02",
        end: "2025-12-31",
        horizon: 1,
        weighting_scheme: "quantile_ls",
        quantile: 0.2,
        commission_bps: 1,
        slippage_bps: 5,
        data_coverage: { first_date: "2020-01-02", last_date: "2025-12-31", observations: 100, symbols: 2, warnings: [] },
        data_revision: "cache-1",
        metrics: { ic: 0.02, ic_ir: 0.5, gross_sharpe: 1.2, net_sharpe: 1.0, max_drawdown: -0.1, turnover: 0.2, avg_positions: 2, max_position: 0.25, usable: true },
        returns: [{ date: "2025-01-02", gross: 0.01, net: 0.009 }],
        normalized_equity: [{ date: "2025-01-02", value: 1.009 }],
        disclaimer: "Exploratory research only.",
      },
    }]);
    keepFormulaTest.mockResolvedValue({ id: "result-1" });
    render(<FormulaBacktestDrawer open source={SOURCE} inputs={[]} formulas={[]} defaultName="Draft test" onClose={() => undefined} />);

    expect(await screen.findByText("Temporary")).toBeInTheDocument();
    expect(keepFormulaTest).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Keep result" }));
    await waitFor(() => expect(keepFormulaTest).toHaveBeenCalledWith("job-done", { name: "Draft test" }));
  });

  it("offers a Data Sync action when the active universe is incomplete", async () => {
    setup();
    const incomplete = await listUniverses();
    listUniverses.mockResolvedValue(incomplete.map((item: { cache_coverage: { complete: boolean } }) => ({
      ...item,
      cache_coverage: { ...item.cache_coverage, complete: false },
    })));
    const onDataSync = vi.fn();
    render(<FormulaBacktestDrawer open source={SOURCE} inputs={[]} formulas={[]} defaultName="Draft test" onDataSync={onDataSync} onClose={() => undefined} />);

    fireEvent.click(await screen.findByRole("button", { name: "Data Sync" }));
    expect(onDataSync).toHaveBeenCalledWith("sample");
  });
});
