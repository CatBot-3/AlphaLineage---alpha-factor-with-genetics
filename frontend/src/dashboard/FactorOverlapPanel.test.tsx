import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const getRoundOverlap = vi.fn();
const startRoundOverlap = vi.fn();
const getRun = vi.fn();

vi.mock("../api/client", () => ({
  getRoundOverlap: (...args: unknown[]) => getRoundOverlap(...args),
  startRoundOverlap: (...args: unknown[]) => startRoundOverlap(...args),
  getRun: (...args: unknown[]) => getRun(...args),
}));

import type { FactorOverlapReport, OverlapReference } from "../api/types";
import { FactorOverlapPanel } from "./FactorOverlapPanel";

function reference(key: string, corr: number, extra: Partial<OverlapReference> = {}): OverlapReference {
  return {
    key,
    name: key,
    display_name: key.replace("formula:", "").toUpperCase(),
    group: "catalog",
    family: "momentum",
    status: "ok",
    mean_rank_corr: corr,
    abs_mean_rank_corr: Math.abs(corr),
    dates: 900,
    reference_ic: 0.012,
    ...extra,
  };
}

function report(extra: Partial<FactorOverlapReport> = {}): FactorOverlapReport {
  return {
    overlap_version: 1,
    session_id: "s1",
    round_index: 0,
    window: { start: "2010-01-04", end: "2015-12-04", dates: 1490, label: "training window" },
    thresholds: { related: 0.3, near_duplicate: 0.7, top_k: 5 },
    candidate: { ic: 0.021, ic_t: 3.1 },
    residual: {
      ic: 0.004,
      ic_t: 0.8,
      explained_ic: 0.017,
      decomposed_ic: 0.021,
      unique_share: 0.19,
      mean_r_squared: 0.74,
      explained_by: ["formula:anom_momentum_12_1"],
    },
    max_abs_rank_corr: 0.84,
    verdict: "near_duplicate",
    references: [
      reference("formula:anom_momentum_12_1", 0.84),
      reference("formula:ta_roc", -0.41),
      reference("result:abc", 0.1, { group: "saved_results", family: "training", display_name: "Kept factor" }),
      { ...reference("formula:broken", 0), status: "unavailable", error: "unknown operator", mean_rank_corr: null },
    ],
    reference_count: 4,
    measured_count: 3,
    horizon: 1,
    execution: "next_open",
    include_saved_results: true,
    computed_at: "2026-09-17T13:00:00+00:00",
    stale: false,
    stale_reasons: [],
    ...extra,
  };
}

afterEach(() => vi.clearAllMocks());

describe("FactorOverlapPanel", () => {
  it("explains a stored near-duplicate and what was removed", async () => {
    getRoundOverlap.mockResolvedValue(report());
    render(<FactorOverlapPanel sessionId="s1" roundIndex={0} />);

    const verdict = await screen.findByTestId("overlap-verdict");
    expect(verdict).toHaveTextContent("Near-duplicate");
    expect(verdict).toHaveTextContent("Closest: ANOM_MOMENTUM_12_1 (rank correlation 0.84)");
    expect(screen.getByTestId("overlap-unique-share")).toHaveTextContent("19%");
    expect(screen.getByTestId("overlap-unique-ic")).toHaveTextContent("0.004");
    expect(screen.getByText(/Removed before measuring unique IC: ANOM_MOMENTUM_12_1/)).toBeInTheDocument();

    const rows = within(screen.getByTestId("overlap-references")).getAllByRole("row");
    expect(rows[1]).toHaveTextContent("removed");
    expect(rows[2]).toHaveTextContent("-0.41");
    expect(rows[3]).toHaveTextContent("Saved result");
    expect(screen.getByText("1 could not be compared")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Re-check" })).toBeEnabled();
    expect(getRoundOverlap).toHaveBeenCalledWith("s1", 0);
  });

  it("runs a first check, shows progress, and loads the stored result", async () => {
    getRoundOverlap.mockResolvedValueOnce(null).mockResolvedValue(report({ verdict: "novel" }));
    startRoundOverlap.mockResolvedValue({ job_id: "job-1", status: "queued", reused: false });
    getRun
      .mockResolvedValueOnce({
        job_id: "job-1",
        status: "running",
        result: null,
        error: null,
        progress: { phase: "reporting", report_done: 50, report_total: 100 },
      })
      .mockResolvedValue({ job_id: "job-1", status: "done", result: null, error: null });

    render(<FactorOverlapPanel sessionId="s1" roundIndex={2} />);
    const button = await screen.findByRole("button", { name: "Check overlap" });
    expect(screen.getByText(/rediscovers a published anomaly/)).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Include my saved Formula Results"));
    fireEvent.click(button);
    await waitFor(() =>
      expect(startRoundOverlap).toHaveBeenCalledWith("s1", 2, { include_saved_results: false }),
    );
    expect(await screen.findByRole("button", { name: "Comparing… 50%" })).toBeDisabled();
    expect(await screen.findByText("Novel", {}, { timeout: 3000 })).toBeInTheDocument();
  });

  it("warns when the stored check is stale and surfaces job failures", async () => {
    getRoundOverlap.mockResolvedValue(
      report({ stale: true, stale_reasons: ["formulas or saved results changed since this check"] }),
    );
    startRoundOverlap.mockResolvedValue({ job_id: "job-2", status: "queued", reused: false });
    getRun.mockResolvedValue({ job_id: "job-2", status: "failed", result: null, error: "no cached dates" });

    render(<FactorOverlapPanel sessionId="s1" roundIndex={0} />);
    expect(await screen.findByTestId("overlap-stale")).toHaveTextContent("formulas or saved results changed");
    fireEvent.click(screen.getByRole("button", { name: "Re-check" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("no cached dates");
  });
});
