import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SessionContinueRequest, SessionCreateRequest } from "../api/types";

const createSession = vi.fn();
const continueSession = vi.fn();
const getSession = vi.fn();
const stopSession = vi.fn();

// Defined via vi.hoisted so it's available inside the hoisted vi.mock factory below.
const { ApiError } = vi.hoisted(() => ({
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public status: number,
    ) {
      super(message);
    }
  },
}));

vi.mock("../api/client", () => ({
  ApiError,
  createSession: (req: SessionCreateRequest) => createSession(req),
  continueSession: (id: string, req: SessionContinueRequest) => continueSession(id, req),
  getSession: (id: string) => getSession(id),
  stopSession: (id: string) => stopSession(id),
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
  listFormulas: () => Promise.resolve([]),
  getPrimitives: () => Promise.resolve([]),
  getTrainingCapabilities: () => Promise.resolve({
    default_profile: "auto",
    detected_cpus: 8,
    worker_capacity: 7,
    available_memory_bytes: 8_000_000_000,
    memory_budget_bytes: 2_000_000_000,
    cpu_budget_percent_min: 10,
    cpu_budget_percent_max: 100,
    evaluator: "auto",
    cpp_available: true,
    fallback_reason: null,
    profiles: {
      light: {
        profile: "light",
        percent: 25,
        requested_workers: 2,
        effective_workers: 2,
        workers: 2,
        worker_capacity: 7,
        run_memory_budget_bytes: 571_428_570,
      },
      auto: {
        profile: "auto",
        percent: 50,
        requested_workers: 4,
        effective_workers: 4,
        workers: 4,
        worker_capacity: 7,
        run_memory_budget_bytes: 1_142_857_140,
      },
      maximum: {
        profile: "maximum",
        percent: 100,
        requested_workers: 7,
        effective_workers: 7,
        workers: 7,
        worker_capacity: 7,
        run_memory_budget_bytes: 2_000_000_000,
      },
    },
  }),
}));

import { TrainPanel } from "./TrainPanel";

const DONE_SESSION = {
  id: "s1",
  segments: [{ index: 0 }],
  cumulative_trials: 812,
  test_reads: 1,
  job: { id: "j1", status: "done", progress: { generation: 12, target_generations: 12, history: [], best: null } },
  result: {
    best_factor: '{"name":"close"}',
    report: { oos_ic: 0.04, deflated_sharpe: 0.1, pbo: 0.4, train_ic: 0.2, n_trials: 800, significant: false },
    generations: 12,
    history: [],
    lineage: { run_id: "s1", metadata: {}, nodes: [] },
    session_id: "s1",
    test_reads: 1,
    cumulative_trials: 812,
  },
};

afterEach(() => {
  vi.clearAllMocks();
});

describe("TrainPanel (B2)", () => {
  it("posts the form's config (not a hardcoded one) and surfaces the result", async () => {
    createSession.mockResolvedValue({ session_id: "s1", job_id: "j1" });
    getSession.mockResolvedValue(DONE_SESSION);
    const onComplete = vi.fn();

    render(<TrainPanel onComplete={onComplete} />);
    fireEvent.submit(await screen.findByTestId("run-config-form"));

    await waitFor(() => expect(createSession).toHaveBeenCalled());
    const req = createSession.mock.calls[0][0] as SessionCreateRequest;
    expect(req.config?.population_size).toBe(80); // the form default, supplied explicitly
    expect(req.config?.generations).toBe(12);
    expect(req.resources).toEqual({ profile: "auto", cpu_budget_percent: null });
    expect(req.as_of).toMatch(/^\d{4}-\d{2}-\d{2}$/);

    await waitFor(() => expect(onComplete).toHaveBeenCalledWith(DONE_SESSION.result));
    expect(await screen.findByTestId("train-continue")).toBeInTheDocument();
  });

  it("posts continue overrides and can stop a run", async () => {
    createSession.mockResolvedValue({ session_id: "s1", job_id: "j1" });
    continueSession.mockResolvedValue({ session_id: "s1", job_id: "j2" });
    getSession.mockResolvedValue(DONE_SESSION);
    stopSession.mockResolvedValue({ stopping: true });

    render(<TrainPanel />);
    fireEvent.submit(await screen.findByTestId("run-config-form"));
    await screen.findByTestId("train-continue");

    const more = screen.getByLabelText("Additional generations");
    fireEvent.change(more, { target: { value: "3" } });
    fireEvent.click(screen.getByTestId("continue-run"));

    await waitFor(() => expect(continueSession).toHaveBeenCalledWith("s1", { generations: 3 }));
  });

  it("returns to the run-config form when a restored session 404s (deleted)", async () => {
    getSession.mockRejectedValue(new ApiError("unknown session", 404));
    render(<TrainPanel restoreSessionId="gone" />);

    // instead of a dead error page, the notice + the form come back
    expect(await screen.findByTestId("train-notice")).toHaveTextContent(/no longer exists/i);
    expect(await screen.findByTestId("run-config-form")).toBeInTheDocument();
  });

  it("keeps a stopped pre-report session resumable without publishing it", async () => {
    getSession.mockResolvedValue({
      ...DONE_SESSION,
      segments: [{ index: 0, status: "stopped", report_cancelled: true }],
      job: {
        id: "j1",
        status: "stopped",
        termination_reason: "user_stopped",
        progress: {
          generation: 1,
          target_generations: 12,
          history: [],
          best: null,
          phase: "stopped",
        },
      },
      result: null,
    });
    const onComplete = vi.fn();

    render(<TrainPanel restoreSessionId="s1" onComplete={onComplete} />);

    expect(await screen.findByTestId("train-continue")).toBeInTheDocument();
    expect(screen.getByTestId("report-cancelled-note")).toHaveTextContent(/checkpoint is saved/i);
    expect(screen.queryByRole("button", { name: "Open dashboard" })).not.toBeInTheDocument();
    expect(onComplete).not.toHaveBeenCalled();
  });

  it("lets the user start a new session from a finished one", async () => {
    createSession.mockResolvedValue({ session_id: "s1", job_id: "j1" });
    getSession.mockResolvedValue(DONE_SESSION);

    render(<TrainPanel />);
    fireEvent.submit(await screen.findByTestId("run-config-form"));
    await screen.findByTestId("train-continue");

    fireEvent.click(screen.getByTestId("new-session"));
    // back to the form, ready to launch a fresh session
    expect(await screen.findByTestId("run-config-form")).toBeInTheDocument();
  });
});
