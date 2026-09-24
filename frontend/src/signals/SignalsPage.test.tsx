import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const listSignalSources = vi.fn();
const listUniverses = vi.fn();
const startSignalSnapshot = vi.fn();
const getSignalSnapshotJob = vi.fn();
const getSignalSnapshot = vi.fn();
const getSignalSeries = vi.fn();
vi.mock("./SignalChart", () => ({SignalChart: ({data}: {data: {symbol: string}}) => <div data-testid="stock-chart" data-symbol={data.symbol}/>}));

vi.mock("../api/client", () => ({
  listSignalSources: () => listSignalSources(),
  listUniverses: () => listUniverses(),
  startSignalSnapshot: (body: unknown) => startSignalSnapshot(body),
  getSignalSnapshotJob: (jobId: string) => getSignalSnapshotJob(jobId),
  getSignalSnapshot: (id: string) => getSignalSnapshot(id),
  getSignalSeries: (...args: unknown[]) => getSignalSeries(...args),
  getPortfolioPreview: vi.fn(),
  signalExportUrl: (id: string, kind: string) => `/signals/snapshots/${id}/export?kind=${kind}`,
  createUniverseFolder: vi.fn(),
  deleteUniverseFolder: vi.fn(),
  moveUniverses: vi.fn(),
  updateUniverseFolder: vi.fn(),
}));

import type { SignalSnapshot, SignalSource, UniverseInfo } from "../api/types";
import { SignalsPage, tradeSentence } from "./SignalsPage";

const SOURCES: SignalSource[] = [
  {
    id: "round:s1:0",
    kind: "round",
    name: "Momentum hunt · round 1",
    universe: "builtin-sp500-sector-energy",
    execution: "next_open",
    horizon: 1,
    evidence: "validation",
    created_at: "2026-09-18T10:00:00+00:00",
  },
  {
    id: "result:abc",
    kind: "saved_result",
    name: "Kept reversal",
    universe: "sp500-lite",
    execution: "close",
    evidence: "saved",
    created_at: "2026-09-01T10:00:00+00:00",
  },
];

const UNIVERSES = [
  { name: "builtin-sp500-sector-energy", display_name: "S&P 500 Energy sector", symbol_count: 21 },
  { name: "sp500-lite", display_name: "Popular US stocks sample", symbol_count: 15 },
] as unknown as UniverseInfo[];

function snapshot(extra: Partial<SignalSnapshot> = {}): SignalSnapshot {
  return {
    signals_version: 1,
    as_of: "2026-09-18",
    previous_as_of: "2026-09-17",
    panel_end: "2026-09-18",
    bars_behind_panel: 0,
    execution: "next_open",
    execution_delay: 1,
    ranked_count: 3,
    excluded_count: 1,
    min_names: 5,
    approximate: false,
    rows: [
      {
        symbol: "XOM",
        rank: 1,
        percentile: 100,
        value: 0.0421,
        previous_rank: 3,
        rank_change: 2,
        close: 118.4,
        day_change: 0.0132,
        last_price_date: "2026-09-18",
      },
      {
        symbol: "CVX",
        rank: 2,
        percentile: 66.7,
        value: 0.0118,
        previous_rank: 2,
        rank_change: 0,
        close: 164.02,
        day_change: -0.0045,
        last_price_date: "2026-09-18",
      },
      {
        symbol: "COP",
        rank: 3,
        percentile: 33.3,
        value: -0.00004,
        previous_rank: 1,
        rank_change: -2,
        close: 99.1,
        day_change: 0.0,
        last_price_date: "2026-09-18",
      },
    ],
    excluded: [
      {
        symbol: "APA",
        reason: "stale_prices",
        detail: "cached prices stop 6 trading day(s) before the snapshot",
        last_price_date: "2026-09-10",
        stale_bars: 6,
      },
    ],
    source: "round:s1:0",
    source_name: "Momentum hunt · round 1",
    evidence: "validation",
    universe: "builtin-sp500-sector-energy",
    history_bars: 260,
    effective_lookback_bars: 20,
    computed_at: "2026-09-19T12:00:00+00:00",
    disclaimer: "Not investment advice. Research output only.",
    ...extra,
  };
}

afterEach(() => vi.clearAllMocks());

function setup(result = snapshot({snapshot_id: "revision-1", status: "partial", actionable: true})) {
  listSignalSources.mockResolvedValue(SOURCES);
  listUniverses.mockResolvedValue(UNIVERSES);
  startSignalSnapshot.mockResolvedValue({job_id: "j1", status: "queued"});
  getSignalSnapshotJob.mockResolvedValue({status: "done", result, error: null});
  getSignalSnapshot.mockResolvedValue(result);
  getSignalSeries.mockResolvedValue({snapshot_id: "revision-1", symbol: "XOM", bars: [], signals: []});
}
async function refresh() {
  const button = await screen.findByRole("button", {name: "Refresh & rank"});
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
}
describe("Signals workspace", () => {
  it("does not download on navigation, then refreshes and inspects an immutable calculation", async () => {
    setup(); render(<SignalsPage/>);
    await waitFor(() => expect(screen.getByLabelText("Formula")).toHaveValue(SOURCES[0].id));
    expect(startSignalSnapshot).not.toHaveBeenCalled();
    await refresh();
    expect(await screen.findByText(/Partial ranking/)).toHaveTextContent("3/4 stocks");
    expect(startSignalSnapshot).toHaveBeenCalledWith(expect.objectContaining({source: SOURCES[0].id, data_mode: "refresh"}));
    expect(await screen.findByTestId("stock-chart")).toBeInTheDocument();
    expect(getSignalSeries).toHaveBeenCalledWith("revision-1", "XOM");
    expect(screen.getByRole("link", {name: "Export ranking CSV"})).toHaveAttribute("href", expect.stringContaining("revision-1"));
    fireEvent.click(screen.getByRole("button", {name: "Show exclusions (1)"}));
    expect(screen.getByRole("button", {name: "APA"})).toBeInTheDocument();
    expect(screen.getByText("stale prices")).toBeInTheDocument();
  });
  it("keeps an explicit library handoff and restored chart range", async () => {
    setup(); render(<SignalsPage workspace={{source: "result:abc", universe: "sp500-lite", range: "1Y"}}/>);
    await screen.findByRole("option", {name: /Kept reversal/});
    expect(screen.getByLabelText("Formula")).toHaveValue("result:abc");
    expect(startSignalSnapshot).not.toHaveBeenCalled();
  });
  it("rechecks market freshness on return without downloading or recalculating", async () => {
    const result = snapshot({snapshot_id: "revision-1", stale: false});
    setup(result);
    render(<SignalsPage workspace={{source: SOURCES[0].id, snapshotId: "revision-1"}}/>);
    await screen.findByText(/Partial ranking/);
    getSignalSnapshot.mockResolvedValue({...result, stale: true, current_expected_session: "2026-09-21"});
    fireEvent.focus(window);
    expect(await screen.findByText(/Stale cached calculation/)).toBeInTheDocument();
    expect(screen.getByText(/Expected completed session:/)).toHaveTextContent("2026-09-21");
    expect(startSignalSnapshot).not.toHaveBeenCalled();
  });
  it("does not replace a newly selected stock with a delayed history response", async () => {
    setup();
    let finish!: (value: unknown) => void;
    getSignalSeries.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }))
      .mockResolvedValue({snapshot_id: "revision-1", symbol: "CVX", bars: [], signals: []});
    render(<SignalsPage/>); await refresh();
    await waitFor(() => expect(getSignalSeries).toHaveBeenCalledWith("revision-1", "XOM"));
    fireEvent.click(screen.getByRole("button", {name: "CVX"}));
    await waitFor(() => expect(screen.getByTestId("stock-chart")).toHaveAttribute("data-symbol", "CVX"));
    finish({snapshot_id: "revision-1", symbol: "XOM", bars: [], signals: []});
    await waitFor(() => expect(screen.getByTestId("stock-chart")).toHaveAttribute("data-symbol", "CVX"));
  });
  it("ignores a late submit response after the formula changes", async () => {
    setup(); let finish: (value: unknown) => void = () => {};
    startSignalSnapshot.mockImplementation(() => new Promise(resolve => {finish = resolve;}));
    render(<SignalsPage/>); await refresh();
    fireEvent.change(screen.getByLabelText("Formula"), {target: {value: "result:abc"}});
    finish({job_id: "obsolete"});
    await waitFor(() => expect(screen.getByRole("button", {name: "Refresh & rank"})).toBeEnabled());
    expect(getSignalSnapshotJob).not.toHaveBeenCalled();
    expect(screen.queryByText(/Partial ranking/)).not.toBeInTheDocument();
  });
  it("blocks portfolio and ranking export below the floor but keeps stock inspection", async () => {
    setup(snapshot({snapshot_id: "revision-1", actionable: false, status: "insufficient_coverage", diagnostic_rows: snapshot().rows, rows: []}));
    render(<SignalsPage/>); await refresh();
    expect(await screen.findByText(/Insufficient coverage/)).toBeInTheDocument();
    expect(screen.getByRole("button", {name: "Portfolio preview"})).toBeDisabled();
    expect(screen.queryByRole("link", {name: "Export ranking CSV"})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {name: "CVX"}));
    await waitFor(() => expect(getSignalSeries).toHaveBeenCalledWith("revision-1", "CVX"));
  });
  it("shows provider failures and does not label failed validation as passed", async () => {
    setup(); getSignalSnapshotJob.mockResolvedValue({status: "failed", error: "Provider quota reached"});
    render(<SignalsPage/>); await refresh();
    expect(await screen.findByRole("alert")).toHaveTextContent("Provider quota reached");
    expect(screen.getByText(/passing criteria were not met/)).toBeInTheDocument();
    expect(screen.getByRole("button", {name: "Retry refresh"})).toBeEnabled();
  });
  it("distinguishes a source loading failure from an empty library", async () => {
    setup(); listSignalSources.mockRejectedValue(new Error("Service unavailable"));
    render(<SignalsPage/>);
    expect(await screen.findByRole("alert")).toHaveTextContent("Service unavailable");
    expect(screen.queryByText(/No formulas available/)).not.toBeInTheDocument();
  });
  it("labels same-close timing as a research assumption", () => {
    expect(tradeSentence(snapshot({execution: "close", execution_delay: 0}))).toContain("same-close research assumption");
  });
});
