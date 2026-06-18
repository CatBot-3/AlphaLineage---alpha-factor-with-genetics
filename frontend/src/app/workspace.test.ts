import { describe, expect, it, beforeEach } from "vitest";
import type { RunResult } from "../api/types";
import {
  WORKSPACE_KEY,
  clearLocalWorkspace,
  makeWorkspaceSnapshot,
  readLocalWorkspace,
  writeLocalWorkspace,
} from "./workspace";

const run: RunResult = {
  best_factor: { name: "close" },
  report: {
    oos_ic: 0.1,
    deflated_sharpe: 0.2,
    pbo: 0.3,
    train_ic: 0.4,
    n_trials: 10,
    significant: false,
  },
  generations: 1,
  history: [{ generation: 0, best_fitness: 0.1, mean_fitness: 0.05, best_ic: 0.1 }],
  lineage: { run_id: "r", metadata: {}, nodes: [] },
};

describe("workspace storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("round-trips run, drafts, and selected UI state", () => {
    const snapshot = makeWorkspaceSnapshot({
      run,
      ui: { selectedTab: "genealogy", selectedLineage: 7 },
      universeDraft: {
        name: "my-universe",
        rows: [{ symbol: "aapl", entry: "2020-01-01", exit: "" }],
      },
      operatorDraft: {
        name: "my_op",
        argTypes: ["series"],
        outType: "series",
        nodes: [{ id: "n1", kind: "$arg", argIndex: 0, x: 10, y: 20 }],
        edges: [],
      },
      formulaDraft: {
        name: "momentum",
        display_name: "Momentum",
        description: "Current value minus delayed value.",
        template: "momentum",
        arg_types: ["series", "window"],
        out_type: "series",
        body: { name: "sub", children: [{ name: "$arg", value: 0 }, { name: "close" }] },
      },
    });

    writeLocalWorkspace(snapshot);
    const loaded = readLocalWorkspace();

    expect(loaded?.run?.best_factor).toEqual({ name: "close" });
    expect(loaded?.universes[0].memberships[0].symbol).toBe("AAPL");
    expect(loaded?.operatorDraft?.nodes[0].kind).toBe("$arg");
    expect(loaded?.formulaDraft?.template).toBe("momentum");
    expect(loaded?.ui.selectedLineage).toBe(7);
  });

  it("ignores malformed or cleared snapshots", () => {
    window.localStorage.setItem(WORKSPACE_KEY, '{"version":2}');
    expect(readLocalWorkspace()).toBeNull();

    clearLocalWorkspace();
    expect(readLocalWorkspace()).toBeNull();
  });
});
