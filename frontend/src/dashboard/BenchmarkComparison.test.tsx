import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { BenchmarkDefinition, BenchmarkSeries, DataSyncJob } from "../api/types";
import {
  comparisonChartSeries,
  deriveEquityLevels,
  BenchmarkComparison,
} from "./BenchmarkComparison";

const listBenchmarks = vi.fn();
const getBenchmarkSeries = vi.fn();
const startDataSync = vi.fn();
const getDataSync = vi.fn();

vi.mock("../api/client", () => ({
  listBenchmarks: (...args: unknown[]) => listBenchmarks(...args),
  getBenchmarkSeries: (...args: unknown[]) => getBenchmarkSeries(...args),
  startDataSync: (...args: unknown[]) => startDataSync(...args),
  getDataSync: (...args: unknown[]) => getDataSync(...args),
}));

const definition: BenchmarkDefinition = {
  id: "sp500",
  label: "S&P 500",
  symbol: "^GSPC",
  color: "#8f2d22",
  return_type: "price_return",
  methodology: "S&P 500 price return.",
};

function benchmark(status: BenchmarkSeries["status"]): BenchmarkSeries {
  return {
    ...definition,
    status,
    message: status === "ready" ? null : "No cached benchmark prices cover this holdout.",
    requested_start: "2025-01-02",
    requested_end: "2025-01-06",
    coverage: {
      symbol: "^GSPC",
      cached: status !== "needs_sync",
      rows: status === "needs_sync" ? 0 : 3,
      first_date: status === "needs_sync" ? null : "2025-01-02",
      last_date: status === "needs_sync" ? null : "2025-01-06",
      requested_start: "2025-01-02",
      requested_end: "2025-01-06",
      needs_sync: status !== "ready",
    },
    normalized_equity: status === "ready"
      ? [
          { date: "2025-01-02", value: 1 },
          { date: "2025-01-03", value: 1.01 },
          { date: "2025-01-06", value: 1.03 },
        ]
      : [],
    sync_request: {
      symbols: ["^GSPC"],
      start: "2025-01-02",
      end: "2025-01-07",
      mode: "incremental",
    },
  };
}

const equity = [
  { date: "2025-01-02", value: 1 },
  { date: "2025-01-03", value: 1.02 },
  { date: "2025-01-06", value: 1.04 },
];
const realizedReturns = [
  { signal_date: "2025-01-02", date: "2025-01-03", gross: 0.021, net: 0.02 },
  { signal_date: "2025-01-03", date: "2025-01-06", gross: 0.021, net: 0.019607843 },
];

describe("benchmark comparison", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listBenchmarks.mockResolvedValue([definition]);
  });

  it("derives the canonical net-equity path when a legacy result only has returns", () => {
    expect(deriveEquityLevels([], [
      { date: "2025-01-02", gross: 0.11, net: 0.1 },
      { date: "2025-01-03", gross: null, net: null },
      { date: "2025-01-06", gross: -0.04, net: -0.05 },
    ])).toEqual([
      { date: "Before 2025-01-02", value: 1 },
      { date: "2025-01-02", value: 1.1 },
      { date: "2025-01-03", value: 1.1 },
      { date: "2025-01-06", value: 1.045 },
    ]);
  });

  it("keeps the first stored point as performance for a metrics-only legacy result", () => {
    expect(deriveEquityLevels([
      { date: "2025-01-02", value: 1.1 },
      { date: "2025-01-03", value: 1.21 },
    ], [])).toEqual([
      { date: "Before 2025-01-02", value: 1 },
      { date: "2025-01-02", value: 1.1 },
      { date: "2025-01-03", value: 1.21 },
    ]);
  });

  it("keeps the full factor path and joins a benchmark at its first overlap", () => {
    const series = comparisonChartSeries(
      [
        { date: "2025-01-02", value: 1 },
        { date: "2025-01-03", value: 1.1 },
        { date: "2025-01-04", value: 1.15 },
        { date: "2025-01-06", value: 1.21 },
      ],
      [{
        ...benchmark("ready"),
        normalized_equity: [
          { date: "2025-01-03", value: 2 },
          { date: "2025-01-06", value: 2.1 },
        ],
      }],
    );

    expect(series[0].points).toHaveLength(4);
    expect(series[0].points[0]).toEqual({ x: "2025-01-02", value: 0 });
    expect(series[0].points[3].value).toBeCloseTo(0.21);
    expect(series[1].points[0].x).toBe("2025-01-03");
    expect(series[1].points[0].value).toBeCloseTo(0.1);
    expect(series[1].points[1].value).toBeCloseTo(0.155);
  });

  it("never erases the factor when visible benchmarks have disjoint cache ranges", () => {
    const first = {
      ...benchmark("ready"),
      normalized_equity: [{ date: "2025-01-02", value: 1 }],
    };
    const second = {
      ...benchmark("ready"),
      id: "djia",
      label: "Dow Jones Industrial Average",
      normalized_equity: [{ date: "2025-01-06", value: 1 }],
    };

    const series = comparisonChartSeries(equity, [first, second]);

    expect(series[0].label).toBe("Factor - net after costs");
    expect(series[0].points).toHaveLength(equity.length);
    expect(series.slice(1).map((item) => item.label)).toEqual([
      "S&P 500 - index price return",
      "Dow Jones Industrial Average - index price return",
    ]);
  });

  it("shows a missing-cache state and only syncs after the explicit action", async () => {
    const user = userEvent.setup();
    getBenchmarkSeries
      .mockResolvedValueOnce(benchmark("needs_sync"))
      .mockResolvedValueOnce(benchmark("ready"));
    startDataSync.mockResolvedValue({ job_id: "sync-1", status: "queued" });
    getDataSync.mockResolvedValue({
      job_id: "sync-1",
      status: "done",
      result: null,
      error: null,
    } satisfies Partial<DataSyncJob>);

    render(<BenchmarkComparison enabled equity={equity} returns={realizedReturns} />);
    const toggle = await screen.findByRole("checkbox", { name: /S&P 500/ });
    await user.click(toggle);

    expect(await screen.findByText(/No cached benchmark prices/)).toBeInTheDocument();
    expect(startDataSync).not.toHaveBeenCalled();
    expect(getBenchmarkSeries).toHaveBeenCalledWith(
      "sp500",
      "2025-01-02",
      "2025-01-06",
    );

    await user.click(screen.getByRole("button", { name: "Sync data" }));
    await waitFor(() => expect(startDataSync).toHaveBeenCalledWith(
      expect.objectContaining({ symbols: ["^GSPC"], mode: "incremental" }),
    ));
    expect(await screen.findByText("S&P 500 - index price return")).toBeInTheDocument();
    expect(screen.getByText(/first cached overlap/)).toBeInTheDocument();

    await user.click(toggle);
    expect(screen.queryByText("S&P 500 - index price return")).not.toBeInTheDocument();
  });

  it("keeps benchmarks unavailable for legacy results without realization dates", () => {
    render(<BenchmarkComparison enabled equity={equity} returns={[]} />);

    expect(screen.getByText(/legacy result has no signal-to-realization dates/i))
      .toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /S&P 500/ })).not.toBeInTheDocument();
    expect(listBenchmarks).not.toHaveBeenCalled();
  });

  it("ignores a cached-series response after its benchmark is hidden", async () => {
    const user = userEvent.setup();
    let resolveSeries!: (value: BenchmarkSeries) => void;
    getBenchmarkSeries.mockReturnValue(new Promise((resolve) => {
      resolveSeries = resolve;
    }));

    render(<BenchmarkComparison enabled equity={equity} returns={realizedReturns} />);
    const toggle = await screen.findByRole("checkbox", { name: /S&P 500/ });
    await user.click(toggle);
    await user.click(toggle);
    resolveSeries(benchmark("ready"));

    await waitFor(() => expect(toggle).not.toBeChecked());
    expect(screen.queryByText("S&P 500 - index price return")).not.toBeInTheDocument();
  });

  it("ignores an old run response after the displayed date range changes", async () => {
    const user = userEvent.setup();
    let resolveOld!: (value: BenchmarkSeries) => void;
    getBenchmarkSeries.mockReturnValue(new Promise((resolve) => {
      resolveOld = resolve;
    }));
    const { rerender } = render(
      <BenchmarkComparison enabled equity={equity} returns={realizedReturns} />,
    );
    await user.click(await screen.findByRole("checkbox", { name: /S&P 500/ }));

    const nextEquity = [
      { date: "2025-02-03", value: 1 },
      { date: "2025-02-04", value: 1.01 },
    ];
    const nextReturns = [
      {
        signal_date: "2025-02-03",
        date: "2025-02-04",
        gross: 0.011,
        net: 0.01,
      },
    ];
    rerender(
      <BenchmarkComparison enabled equity={nextEquity} returns={nextReturns} />,
    );
    resolveOld(benchmark("ready"));

    const nextToggle = await screen.findByRole("checkbox", { name: /S&P 500/ });
    await waitFor(() => expect(nextToggle).not.toBeChecked());
    expect(screen.queryByText("S&P 500 - index price return")).not.toBeInTheDocument();
  });
});
