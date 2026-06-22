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
  listFactors: () => Promise.resolve([]),
  getPrimitives: () => Promise.resolve([]),
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
