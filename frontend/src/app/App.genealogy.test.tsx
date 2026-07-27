import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./mode", () => ({ getAppMode: () => "app", modeLabel: () => "Local Backend" }));

const getSession = vi.fn();
const getSessionLineage = vi.fn();
const getSessionRound = vi.fn();
const listSessionRounds = vi.fn();
const listSessionFinalizations = vi.fn();
const getSessionFinalization = vi.fn();
const finalizeSessionRound = vi.fn();
const getRun = vi.fn();
const createSession = vi.fn();
const listStrategyComparisons = vi.fn();
const getStrategyComparison = vi.fn();
const listFinalizationPlans = vi.fn();
const createFinalizationPlan = vi.fn();

// Session result with the lineage STRIPPED (as GET /sessions/{id} returns it).
const RESULT = {
  best_factor: '{"name":"close"}',
  report: { oos_ic: 0.04, deflated_sharpe: 0.1, pbo: 0.4, train_ic: 0.2, n_trials: 8, significant: false },
  generations: 2,
  history: [],
  session_id: "s1",
  test_reads: 1,
  cumulative_trials: 10,
};

const LINEAGE = {
  run_id: "s1",
  metadata: {},
  nodes: [
    { id: 0, generation: 0, op: "init", parents: [], tree: { name: "close" }, fitness: 0.1 },
    { id: 1, generation: 1, op: "elite", parents: [0], tree: { name: "close" }, fitness: 0.2 },
  ],
};

vi.mock("../api/client", () => ({
  ApiError: class extends Error {},
  createSession: (r: unknown) => createSession(r),
  getSession: (id: string) => getSession(id),
  getSessionLineage: (id: string) => getSessionLineage(id),
  getSessionRound: (id: string, round: number) => getSessionRound(id, round),
  listSessionRounds: (id: string) => listSessionRounds(id),
  listSessionFinalizations: (id: string) => listSessionFinalizations(id),
  getSessionFinalization: (id: string, evaluation: string) =>
    getSessionFinalization(id, evaluation),
  finalizeSessionRound: (
    id: string,
    round: number,
    confirm: boolean,
    strategyPlanId?: string,
  ) => finalizeSessionRound(id, round, confirm, strategyPlanId),
  getRun: (id: string) => getRun(id),
  listSessions: () => Promise.resolve([]),
  continueSession: vi.fn(),
  stopSession: vi.fn(),
  saveFactor: vi.fn(),
  saveWorkspace: vi.fn(),
  listWorkspaces: () => Promise.resolve([]),
  getWorkspace: vi.fn(),
  shutdown: vi.fn(),
  getPrimitives: () => Promise.resolve([]),
  listFormulas: () => Promise.resolve([]),
  addFormula: vi.fn(),
  deleteFormula: vi.fn(),
  defineUniverse: vi.fn(),
  getUniverse: vi.fn(),
  updateUniverse: vi.fn(),
  deleteUniverse: vi.fn(),
  searchSymbols: () => Promise.resolve([]),
  validateSymbol: vi.fn(),
  getDataCoverage: () => Promise.resolve([]),
  startDataSync: vi.fn(),
  getDataSync: vi.fn(),
  listUniverses: () => Promise.resolve([]),
  getUniverseCoverage: () =>
    Promise.resolve({
      as_of: "2026-07-15",
      eligible_symbols: ["AAPL"],
      cached_symbols: ["AAPL"],
      missing_symbols: [],
      incomplete_symbols: [],
      complete: true,
    }),
  listFormulaResults: () => Promise.resolve([]),
  getSettings: () => Promise.resolve({ factors_dir: "/d", tiingo_api_key_set: false, evaluator: "auto", cpp_available: false }),
  getDataUsage: () => Promise.resolve([]),
  listStrategyComparisons: (...args: unknown[]) => listStrategyComparisons(...args),
  listFinalizationPlans: (...args: unknown[]) => listFinalizationPlans(...args),
  createStrategyComparison: vi.fn(),
  getStrategyComparison: (...args: unknown[]) => getStrategyComparison(...args),
  createFinalizationPlan: (...args: unknown[]) => createFinalizationPlan(...args),
  putSettings: vi.fn(),
  clearData: vi.fn(),
}));

import { App } from "./App";

beforeEach(() => {
  window.localStorage.clear();
  listSessionRounds.mockResolvedValue([]);
  listSessionFinalizations.mockResolvedValue([]);
  listStrategyComparisons.mockResolvedValue([]);
  listFinalizationPlans.mockResolvedValue([]);
});
afterEach(() => vi.clearAllMocks());

describe("Genealogy after a session run (F2 - no blank page)", () => {
  it("fetches the stripped lineage and renders the generation list", async () => {
    createSession.mockResolvedValue({ session_id: "s1", job_id: "j1" });
    getSession.mockResolvedValue({
      id: "s1",
      segments: [{ index: 0 }],
      cumulative_trials: 10,
      test_reads: 1,
      job: { id: "j1", status: "done", progress: null },
      result: RESULT, // no lineage here
    });
    getSessionLineage.mockResolvedValue(LINEAGE);

    render(<App />);
    fireEvent.submit(await screen.findByTestId("run-config-form"));

    // completion triggers a lineage fetch and navigates to the dashboard
    await waitFor(() => expect(getSessionLineage).toHaveBeenCalledWith("s1"));
    await screen.findByTestId("primary-metric");
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1, name: "Metrics" })).toBeInTheDocument();

    // the Genealogy tab now renders the grouped list (previously a blank page)
    fireEvent.click(screen.getByRole("button", { name: "Genealogy" }));
    expect(await screen.findByTestId("generation-list")).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1, name: "Genealogy" })).toBeInTheDocument();
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument();
  });

  it("keeps polling after the Train tab unmounts", async () => {
    createSession.mockResolvedValue({ session_id: "s1", job_id: "j1" });
    let finishRequest: ((value: unknown) => void) | undefined;
    getSession.mockImplementation(
      () =>
        new Promise((resolve) => {
          finishRequest = resolve;
        }),
    );
    getSessionLineage.mockResolvedValue(LINEAGE);

    render(<App />);
    fireEvent.submit(await screen.findByTestId("run-config-form"));
    await waitFor(() => expect(getSession).toHaveBeenCalledWith("s1"));
    await waitFor(() => {
      const stored = JSON.parse(
        window.localStorage.getItem("alphalineage:workspace:v1") ?? "{}",
      );
      expect(stored.ui.sessionId).toBe("s1");
    });

    fireEvent.click(screen.getByRole("button", { name: "Library" }));
    expect(await screen.findByTestId("library-panel")).toBeInTheDocument();

    finishRequest?.({
      id: "s1",
      segments: [{ index: 0 }],
      cumulative_trials: 10,
      test_reads: 1,
      job: { id: "j1", status: "done", progress: null },
      result: RESULT,
    });

    expect(await screen.findByTestId("primary-metric")).toBeInTheDocument();
    expect(getSessionLineage).toHaveBeenCalledWith("s1");
  });

  it("keeps validation rounds report-free until explicit finalization", async () => {
    const selection = {
      validated: true,
      first_seen: 0,
      polarity: 1,
      training_fitness: 0.05,
      training_metrics: { signed_ic: 0.05, oriented_ic: 0.05 },
      validation_fitness: 0.04,
      validation_metrics: { signed_ic: 0.04, valid_dates: 210 },
      folds: [
        { index: 0, valid_dates: 70, oriented_ic: 0.03, positive: true },
        { index: 1, valid_dates: 70, oriented_ic: 0.04, positive: true },
        { index: 2, valid_dates: 70, oriented_ic: -0.01, positive: false },
      ],
      positive_folds: 2,
      median_oriented_ic: 0.03,
    };
    const validationResult = {
      ...RESULT,
      report: null,
      round_index: 0,
      validation_only: true,
      evidence_status: "validation_only",
      selection,
    };
    const roundSummary = {
      index: 0,
      segment_index: 0,
      report_available: false,
      finalization_available: false,
      test_read_index: null,
      evidence_status: "validation_only",
      status: "done",
      requested_generations: 2,
      gen_start: 0,
      gen_end: 2,
    };
    createSession.mockResolvedValue({ session_id: "s1", job_id: "j1" });
    getSession.mockImplementation(() =>
      Promise.resolve({
        id: "s1",
        segments: [{ index: 0, status: "done" }],
        rounds: [roundSummary],
        cumulative_trials: 10,
        test_reads: finalized ? 1 : 0,
        session_holdout_reads: finalized ? 1 : 0,
        job: { id: "j1", status: "done", progress: null },
        finalization_job: finalized
          ? {
              id: "j-final",
              status: "done",
              progress: null,
              evaluation_id: "eval-1",
              round_index: 0,
            }
          : null,
        result: validationResult,
      }),
    );
    getSessionRound.mockResolvedValue(validationResult);
    getSessionLineage.mockResolvedValue(LINEAGE);
    listSessionRounds.mockResolvedValue([roundSummary]);

    let finalized = false;
    listStrategyComparisons.mockResolvedValue([
      {
        comparison_id: "cmp-1",
        round_index: 0,
        status: "completed",
        result_available: true,
        strategies: [],
      },
    ]);
    getStrategyComparison.mockResolvedValue({
      comparison_id: "cmp-1",
      round_index: 0,
      status: "completed",
      strategies: [],
      strategy_results: [
        {
          strategy_id: "quantile_ls_20",
          spec: { id: "quantile_ls_20", scheme: "quantile_ls", quantile: 0.2 },
          eligible: true,
          validation_backtest: {
            start: "2024-01-01",
            end: "2024-06-01",
            observations: 100,
            metrics: {
              signed_ic: 0.03,
              mean_abs_ic: 0.04,
              ic_ir: 0.5,
              gross_sharpe: 1,
              net_sharpe: 0.9,
              max_drawdown: -0.1,
              turnover: 0.2,
              avg_gross: 1,
              avg_positions: 10,
              max_position: 0.1,
              usable: true,
            },
            returns: [],
            normalized_equity: [
              { date: "2024-01-01", value: 1 },
              { date: "2024-06-01", value: 1.1 },
            ],
            portfolio_health: {
              eligible_dates: 100,
              calendar_observations: 100,
              active_observations: 90,
              two_sided_observations: 90,
              exposure_coverage: 0.9,
              flat_factor_dates: 10,
              valid: true,
              reason: null,
            },
          },
        },
      ],
    });
    createFinalizationPlan.mockResolvedValue({
      strategy_plan_id: "plan-1",
      session_id: "s1",
      round_index: 0,
      comparison_id: "cmp-1",
      primary_strategy_id: "quantile_ls_20",
      strategies: [{ id: "quantile_ls_20", scheme: "quantile_ls", quantile: 0.2 }],
    });
    listSessionFinalizations.mockImplementation(() =>
      Promise.resolve(
        finalized
          ? [
              {
                evaluation_id: "eval-1",
                round_index: 0,
                status: "done",
                evidence_status: "locked_first_read",
                same_holdout_read_index: 1,
                session_holdout_reads: 1,
                holdout_fingerprint: "holdout-a",
                report_available: true,
              },
            ]
          : [],
      ),
    );
    finalizeSessionRound.mockImplementation(() => {
      finalized = true;
      return Promise.resolve({
        session_id: "s1",
        round_index: 0,
        evaluation_id: "eval-1",
        job_id: "j-final",
        status: "queued",
      });
    });
    getRun.mockResolvedValue({ job_id: "j-final", status: "done", result: null, error: null });
    getSessionFinalization.mockResolvedValue({
      evaluation_id: "eval-1",
      round_index: 0,
      status: "done",
      evidence_status: "locked_first_read",
      same_holdout_read_index: 1,
      session_holdout_reads: 1,
      holdout_fingerprint: "holdout-a",
      report_available: true,
      best_factor: '{"name":"close"}',
      report: RESULT.report,
      selection,
    });

    render(<App />);
    fireEvent.submit(await screen.findByTestId("run-config-form"));
    expect(await screen.findByTestId("strategy-comparison")).toBeInTheDocument();
    expect(screen.queryByTestId("holdout-performance")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Pin primary strategy" }));
    await waitFor(() =>
      expect(createFinalizationPlan).toHaveBeenCalledWith(
        "s1",
        0,
        "cmp-1",
        "quantile_ls_20",
      ),
    );
    fireEvent.click(screen.getByTestId("finalize-round"));
    expect(await screen.findByTestId("holdout-performance")).toBeInTheDocument();
    expect(finalizeSessionRound).toHaveBeenCalledWith("s1", 0, false, "plan-1");
    expect(getSessionFinalization).toHaveBeenCalledWith("s1", "eval-1");
  });
});
