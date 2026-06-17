import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("./mode", () => ({ getAppMode: () => "app", modeLabel: () => "Local Backend" }));

const getSession = vi.fn();
const getSessionLineage = vi.fn();
const createSession = vi.fn();

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
  continueSession: vi.fn(),
  stopSession: vi.fn(),
  saveFactor: vi.fn(),
  saveWorkspace: vi.fn(),
  listWorkspaces: () => Promise.resolve([]),
  getWorkspace: vi.fn(),
  shutdown: vi.fn(),
  listUniverses: () => Promise.resolve([]),
  listFactors: () => Promise.resolve([]),
  getSettings: () => Promise.resolve({ factors_dir: "/d", tiingo_api_key_set: false, evaluator: "auto", cpp_available: false }),
  getDataUsage: () => Promise.resolve([]),
  putSettings: vi.fn(),
  clearData: vi.fn(),
}));

import { App } from "./App";

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

    // the Genealogy tab now renders the grouped list (previously a blank page)
    fireEvent.click(screen.getByRole("button", { name: "Genealogy" }));
    expect(await screen.findByTestId("generation-list")).toBeInTheDocument();
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument();
  });
});
