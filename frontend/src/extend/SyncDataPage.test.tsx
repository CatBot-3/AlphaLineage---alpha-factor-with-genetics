import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const getDataCoverage = vi.fn();
const getDataSync = vi.fn();
const startDataSync = vi.fn();

vi.mock("../api/client", () => ({
  getDataCoverage: (...args: unknown[]) => getDataCoverage(...args),
  getDataSync: (jobId: string) => getDataSync(jobId),
  startDataSync: (payload: unknown) => startDataSync(payload),
}));

import { SyncDataPage } from "./SyncDataPage";

const ROWS = [{ symbol: "AAPL", entry: "2020-01-01", exit: "" }];

afterEach(() => {
  vi.clearAllMocks();
});

describe("SyncDataPage", () => {
  it("shows sync controls immediately (no longer collapsed) and can choose refresh mode", async () => {
    getDataCoverage.mockResolvedValue([]);
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
    getDataCoverage.mockResolvedValue([]);
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
});
