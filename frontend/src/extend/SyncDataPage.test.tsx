import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getDataCoverage = vi.fn();
const getDataSync = vi.fn();
const getUniverse = vi.fn();
const listDataSyncs = vi.fn();
const startDataSync = vi.fn();
const stopDataSync = vi.fn();

vi.mock("../api/client", () => ({
  getDataCoverage: (...args: unknown[]) => getDataCoverage(...args),
  getDataSync: (jobId: string) => getDataSync(jobId),
  getUniverse: (name: string) => getUniverse(name),
  listDataSyncs: (...args: unknown[]) => listDataSyncs(...args),
  startDataSync: (payload: unknown) => startDataSync(payload),
  stopDataSync: (jobId: string) => stopDataSync(jobId),
}));

import type { UniverseInfo } from "../api/types";
import { UniverseDataSync } from "./SyncDataPage";

const ROWS = [{ symbol: "AAPL", entry: "2020-01-01", exit: "" }];
const SAVED_UNIVERSE: UniverseInfo = {
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
};

beforeEach(() => {
  getDataCoverage.mockResolvedValue([]);
  getUniverse.mockResolvedValue(SAVED_UNIVERSE);
  listDataSyncs.mockResolvedValue([]);
  stopDataSync.mockResolvedValue({ stopping: true });
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("UniverseDataSync", () => {
  it("syncs the current draft with the visible date range and refresh mode", async () => {
    startDataSync.mockResolvedValue({ job_id: "sync-1", status: "queued", reused: false });
    getDataSync.mockResolvedValue({
      job_id: "sync-1",
      status: "done",
      result: {
        mode: "refresh",
        start: "2020-01-01",
        end: null,
        results: [],
        failed_count: 0,
        succeeded_count: 0,
      },
      error: null,
      progress: { done: 1, total: 1, current_symbol: "AAPL" },
    });

    render(<UniverseDataSync rows={ROWS} recommendedStart="2020-01-01" />);

    const incremental = screen.getByLabelText("Incremental merge");
    expect(incremental).toBeChecked();
    fireEvent.click(incremental);
    fireEvent.click(screen.getByTestId("sync-universe"));

    await waitFor(() => expect(startDataSync).toHaveBeenCalledWith({
      symbols: ["AAPL"],
      start: "2020-01-01",
      end: undefined,
      mode: "refresh",
    }));
  });

  it("syncs a loaded universe by pinned identity and refreshes its readiness", async () => {
    const refreshed = {
      ...SAVED_UNIVERSE,
      cache_coverage: {
        as_of: "2026-07-16",
        eligible_symbols: ["AAPL", "BRK.B"],
        cached_symbols: ["AAPL", "BRK.B"],
        missing_symbols: [],
        incomplete_symbols: [],
        complete: true,
      },
    };
    getUniverse.mockResolvedValue(refreshed);
    startDataSync.mockResolvedValue({ job_id: "sync-index", status: "queued", reused: false });
    getDataSync.mockResolvedValue({
      job_id: "sync-index",
      status: "done",
      request: {
        universe: SAVED_UNIVERSE.name,
        start: "2000-01-01",
        mode: "incremental",
      },
      result: {
        universe: SAVED_UNIVERSE.name,
        mode: "incremental",
        start: "2000-01-01",
        failed_count: 0,
        succeeded_count: 1,
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
    const onUniverseRefresh = vi.fn();

    render(
      <UniverseDataSync
        rows={ROWS}
        universeName={SAVED_UNIVERSE.name}
        universe={SAVED_UNIVERSE}
        recommendedStart="2000-01-01"
        onUniverseRefresh={onUniverseRefresh}
      />,
    );
    expect(screen.getByTestId("sync-universe-definition")).toHaveTextContent("sha256:index");
    fireEvent.click(screen.getByTestId("sync-universe"));

    await waitFor(() => expect(startDataSync).toHaveBeenCalledWith({
      universe: SAVED_UNIVERSE.name,
      start: "2000-01-01",
      end: undefined,
      mode: "incremental",
    }));
    await waitFor(() => expect(onUniverseRefresh).toHaveBeenCalledWith(refreshed));
    fireEvent.click(screen.getByRole("button", { name: /Latest sync result/ }));
    expect(await screen.findByTestId("sync-results")).toHaveTextContent("Provider symbol: BRK-B");
  });

  it("reattaches by pinned fingerprint and can stop the active job", async () => {
    const active = {
      job_id: "sync-active",
      status: "running",
      request: {
        universe: "renamed-index",
        start: "2020-01-01",
        mode: "incremental",
        resolved_symbols: ["AAPL", "BRK.B"],
      },
      universe_definition: {
        name: "builtin-sp500-current",
        requested_name: "renamed-index",
        source: "bundled",
        mode: "static_snapshot",
        definition: SAVED_UNIVERSE.definition,
        fingerprint: "sha256:index",
        provenance: { provider: "Wikipedia" },
        aliases: { "BRK.B": "BRK-B" },
      },
      result: null,
      error: null,
      progress: { done: 1, total: 2, current_symbol: "BRK.B" },
    };
    listDataSyncs.mockResolvedValue([active]);
    getDataSync.mockResolvedValue({ ...active, status: "stopped" });

    render(
      <UniverseDataSync
        rows={ROWS}
        universeName={SAVED_UNIVERSE.name}
        universe={SAVED_UNIVERSE}
      />,
    );

    const jobs = await screen.findByTestId("active-sync-jobs");
    expect(within(jobs).getByRole("button", { name: "Attached" })).toBeInTheDocument();
    fireEvent.click(within(jobs).getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(stopDataSync).toHaveBeenCalledWith("sync-active"));
  });

  it("labels a terminal batch with partial failures instead of claiming success", async () => {
    startDataSync.mockResolvedValue({ job_id: "sync-partial", status: "queued", reused: false });
    getDataSync.mockResolvedValue({
      job_id: "sync-partial",
      status: "done",
      result: {
        mode: "incremental",
        start: "2020-01-01",
        failed_count: 1,
        succeeded_count: 1,
        results: [
          { symbol: "AAPL", status: "fetched", rows_fetched: 10, rows_cached: 10 },
          { symbol: "MSFT", status: "failed", rows_fetched: 0, rows_cached: 0, error: "timeout" },
        ],
      },
      error: null,
    });

    render(<UniverseDataSync rows={ROWS} />);
    fireEvent.click(screen.getByTestId("sync-universe"));
    expect(await screen.findByText(/Sync completed with 1 failed symbol/)).toBeInTheDocument();
    expect(screen.getByText("Completed with 1 failure")).toBeInTheDocument();
  });

  it("explains a provider allowance stop and how many symbols were never requested", async () => {
    startDataSync.mockResolvedValue({ job_id: "sync-quota", status: "queued", reused: false });
    getDataSync.mockResolvedValue({
      job_id: "sync-quota",
      status: "done",
      result: {
        mode: "incremental",
        start: "2020-01-01",
        failed_count: 0,
        succeeded_count: 1,
        termination_reason: "quota_exceeded",
        quota: { provider: "tiingo", scope: "hourly", message: "hourly allocation" },
        not_attempted: ["MSFT", "NVDA"],
        results: [
          { symbol: "AAPL", status: "fetched", rows_fetched: 10, rows_cached: 10 },
          {
            symbol: "AMZN",
            status: "quota_exceeded",
            rows_fetched: 0,
            rows_cached: 0,
            error: "Tiingo hourly allowance exhausted",
            quota_scope: "hourly",
          },
        ],
      },
      error: null,
    });

    render(<UniverseDataSync rows={ROWS} />);
    fireEvent.click(screen.getByTestId("sync-universe"));
    expect(await screen.findByText("Stopped: hourly allowance used")).toBeInTheDocument();
    expect(screen.getByTestId("sync-quota-stop")).toHaveTextContent(
      "The tiingo hourly allowance is used up, so the sync stopped instead of spending more requests (2 symbols not attempted).",
    );
    expect(screen.getByText("1 completed · 0 failed · 2 not attempted")).toBeInTheDocument();
  });

  it("continues polling beyond the former 60-second cap", async () => {
    vi.useFakeTimers();
    try {
      startDataSync.mockResolvedValue({ job_id: "sync-long", status: "queued", reused: false });
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

      render(<UniverseDataSync rows={ROWS} />);
      fireEvent.click(screen.getByTestId("sync-universe"));
      await act(async () => {
        await Promise.resolve();
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(61_000);
      });

      expect(polls).toBeGreaterThan(120);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not publish a saved-universe refresh after the panel unmounts", async () => {
    let resolveUniverse!: (value: UniverseInfo) => void;
    getUniverse.mockReturnValue(new Promise((resolve) => {
      resolveUniverse = resolve;
    }));
    const onUniverseRefresh = vi.fn();
    const view = render(
      <UniverseDataSync
        rows={ROWS}
        universeName={SAVED_UNIVERSE.name}
        universe={SAVED_UNIVERSE}
        onUniverseRefresh={onUniverseRefresh}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Refresh coverage" }));
    await waitFor(() => expect(getUniverse).toHaveBeenCalledWith(SAVED_UNIVERSE.name));
    view.unmount();
    await act(async () => {
      resolveUniverse(SAVED_UNIVERSE);
    });

    expect(onUniverseRefresh).not.toHaveBeenCalled();
  });

  it("abandons a pending sync start when the target universe changes", async () => {
    let resolveStart!: (value: { job_id: string; status: "queued"; reused: boolean }) => void;
    startDataSync.mockReturnValue(new Promise((resolve) => {
      resolveStart = resolve;
    }));
    const onPullProgress = vi.fn();
    const view = render(
      <UniverseDataSync
        rows={ROWS}
        universeName={SAVED_UNIVERSE.name}
        universe={SAVED_UNIVERSE}
        onPullProgress={onPullProgress}
      />,
    );

    fireEvent.click(screen.getByTestId("sync-universe"));
    await waitFor(() => expect(startDataSync).toHaveBeenCalledTimes(1));
    const replacement = {
      ...SAVED_UNIVERSE,
      name: "builtin-djia-current",
      display_name: "Dow Jones Industrial Average static snapshot",
      fingerprint: "sha256:djia",
    };
    view.rerender(
      <UniverseDataSync
        rows={ROWS}
        universeName={replacement.name}
        universe={replacement}
        onPullProgress={onPullProgress}
      />,
    );
    await act(async () => {
      resolveStart({ job_id: "stale-sync", status: "queued", reused: false });
    });

    expect(getDataSync).not.toHaveBeenCalledWith("stale-sync");
    expect(onPullProgress.mock.calls.some(([value]) => value !== null)).toBe(false);
  });

  it("does not publish a fetched sync job after the panel unmounts", async () => {
    startDataSync.mockResolvedValue({ job_id: "pending-job", status: "queued", reused: false });
    let resolveJob!: (value: {
      job_id: string;
      status: "running";
      result: null;
      error: null;
      progress: { done: number; total: number; current_symbol: string };
    }) => void;
    getDataSync.mockReturnValue(new Promise((resolve) => {
      resolveJob = resolve;
    }));
    const onPullProgress = vi.fn();
    const view = render(
      <UniverseDataSync
        rows={ROWS}
        universeName={SAVED_UNIVERSE.name}
        universe={SAVED_UNIVERSE}
        onPullProgress={onPullProgress}
      />,
    );

    fireEvent.click(screen.getByTestId("sync-universe"));
    await waitFor(() => expect(getDataSync).toHaveBeenCalledWith("pending-job"));
    view.unmount();
    await act(async () => {
      resolveJob({
        job_id: "pending-job",
        status: "running",
        result: null,
        error: null,
        progress: { done: 1, total: 2, current_symbol: "AAPL" },
      });
    });

    expect(onPullProgress.mock.calls.some(([value]) => value !== null)).toBe(false);
  });
});
