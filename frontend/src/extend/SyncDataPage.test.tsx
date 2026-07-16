import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getDataCoverage = vi.fn();
const getDataSync = vi.fn();
const getUniverse = vi.fn();
const listDataSyncs = vi.fn();
const listUniverses = vi.fn();
const startDataSync = vi.fn();
const stopDataSync = vi.fn();

vi.mock("../api/client", () => ({
  getDataCoverage: (...args: unknown[]) => getDataCoverage(...args),
  getDataSync: (jobId: string) => getDataSync(jobId),
  getUniverse: (name: string) => getUniverse(name),
  listDataSyncs: () => listDataSyncs(),
  listUniverses: () => listUniverses(),
  startDataSync: (payload: unknown) => startDataSync(payload),
  stopDataSync: (jobId: string) => stopDataSync(jobId),
}));

import { SyncDataPage } from "./SyncDataPage";

const ROWS = [{ symbol: "AAPL", entry: "2020-01-01", exit: "" }];

beforeEach(() => {
  getDataCoverage.mockResolvedValue([]);
  listDataSyncs.mockResolvedValue([]);
  listUniverses.mockResolvedValue([]);
  stopDataSync.mockResolvedValue({ stopping: true });
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("SyncDataPage", () => {
  it("shows sync controls immediately (no longer collapsed) and can choose refresh mode", async () => {
    startDataSync.mockResolvedValue({ job_id: "sync-1", status: "queued" });
    getDataSync.mockResolvedValue({
      job_id: "sync-1",
      status: "done",
      result: { mode: "refresh", start: "2020-01-01", end: null, results: [] },
      error: null,
      progress: { done: 1, total: 1, current_symbol: "AAPL" },
    });

    render(<SyncDataPage rows={ROWS} />);

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

  it("forwards live progress to onPullProgress and clears it when the page unmounts", async () => {
    startDataSync.mockResolvedValue({ job_id: "sync-1", status: "queued" });
    getDataSync.mockResolvedValue({
      job_id: "sync-1",
      status: "done",
      result: { mode: "incremental", start: "2020-01-01", end: null, results: [] },
      error: null,
      progress: { done: 1, total: 1, current_symbol: "AAPL" },
    });
    const onPullProgress = vi.fn();

    const { unmount } = render(<SyncDataPage rows={ROWS} onPullProgress={onPullProgress} />);
    fireEvent.click(screen.getByTestId("sync-universe"));

    await waitFor(() =>
      expect(onPullProgress).toHaveBeenCalledWith({ done: 1, total: 1, current_symbol: "AAPL" }),
    );
    onPullProgress.mockClear();
    unmount();
    expect(onPullProgress).toHaveBeenCalledWith(null);
  });

  it("syncs a saved universe by identity and shows its aliases and definition", async () => {
    listUniverses.mockResolvedValue([{
      name: "builtin-sp500-current",
      display_name: "S&P 500 static snapshot",
      source: "bundled",
      mode: "static_snapshot",
      symbols: [],
      memberships: [],
      definition: {
        id: "builtin-sp500-current",
        display_name: "S&P 500 static snapshot",
        snapshot_date: "2026-07-14",
        interval_semantics: "Static membership",
        member_count: 500,
      },
      fingerprint: "sha256:index",
    }]);
    getUniverse.mockResolvedValue({
      name: "builtin-sp500-current",
      display_name: "S&P 500 static snapshot",
      source: "bundled",
      mode: "static_snapshot",
      symbols: ["AAPL", "BRK.B"],
      memberships: [],
      definition: {
        id: "builtin-sp500-current",
        display_name: "S&P 500 static snapshot",
        snapshot_date: "2026-07-14",
        interval_semantics: "Static membership",
        member_count: 500,
      },
      fingerprint: "sha256:index",
      aliases: { "BRK.B": "BRK-B" },
      readiness: {
        membership_ready: true,
        price_ready: false,
        research_ready: false,
        issues: ["Prices missing"],
      },
    });
    startDataSync.mockResolvedValue({ job_id: "sync-index", status: "queued" });
    getDataSync.mockResolvedValue({
      job_id: "sync-index",
      status: "done",
      request: { universe: "builtin-sp500-current", start: "2020-01-01", mode: "incremental" },
      result: {
        universe: "builtin-sp500-current",
        mode: "incremental",
        start: "2020-01-01",
        results: [{
          symbol: "BRK.B",
          provider_symbol: "BRK-B",
          status: "fetched",
          rows_fetched: 10,
          rows_cached: 10,
        }],
      },
      error: null,
      progress: { done: 2, total: 2, current_symbol: "BRK.B" },
    });

    render(<SyncDataPage rows={ROWS} />);
    fireEvent.change(await screen.findByLabelText("Sync target"), {
      target: { value: "builtin-sp500-current" },
    });

    const definition = await screen.findByTestId("sync-universe-definition");
    expect(definition).toHaveTextContent("static snapshot");
    expect(definition).toHaveTextContent("BRK.B → BRK-B");
    fireEvent.click(screen.getByTestId("sync-universe"));

    await waitFor(() => expect(startDataSync).toHaveBeenCalledWith({
      universe: "builtin-sp500-current",
      start: "2020-01-01",
      end: undefined,
      mode: "incremental",
    }));
    expect(await screen.findByTestId("sync-results")).toHaveTextContent("Provider symbol: BRK-B");
  });

  it("lists active jobs and supports reattaching and stopping them", async () => {
    const active = {
      job_id: "sync-active",
      status: "running",
      request: {
        universe: "builtin-sp500-current",
        start: "2020-01-01",
        mode: "incremental",
        resolved_symbols: ["AAPL", "MSFT"],
      },
      result: null,
      error: null,
      progress: { done: 1, total: 2, current_symbol: "MSFT" },
    };
    listDataSyncs.mockResolvedValue([active]);
    getDataSync.mockResolvedValue({ ...active, status: "stopped" });

    render(<SyncDataPage rows={ROWS} />);

    const jobs = await screen.findByTestId("active-sync-jobs");
    expect(jobs).toHaveTextContent("builtin-sp500-current");
    fireEvent.click(within(jobs).getByRole("button", { name: "Reattach" }));
    fireEvent.click(within(jobs).getByRole("button", { name: "Stop" }));

    await waitFor(() => expect(stopDataSync).toHaveBeenCalledWith("sync-active"));
    expect(getDataSync).toHaveBeenCalledWith("sync-active");
  });

  it("continues polling beyond the former 60-second cap", async () => {
    vi.useFakeTimers();
    try {
      startDataSync.mockResolvedValue({ job_id: "sync-long", status: "queued" });
      let polls = 0;
      getDataSync.mockImplementation(async () => {
        polls += 1;
        const done = polls >= 122;
        return {
          job_id: "sync-long",
          status: done ? "done" : "running",
          request: { symbols: ["AAPL"], start: "2020-01-01", mode: "incremental" },
          result: done
            ? { mode: "incremental", start: "2020-01-01", results: [] }
            : null,
          error: null,
          progress: { done: done ? 1 : 0, total: 1, current_symbol: "AAPL" },
        };
      });

      render(<SyncDataPage rows={ROWS} />);
      fireEvent.click(screen.getByTestId("sync-universe"));
      await act(async () => {
        await Promise.resolve();
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(61_000);
      });

      expect(polls).toBeGreaterThan(120);
      expect(screen.getByText("1 draft symbols")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
