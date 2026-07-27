import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const shutdown = vi.fn();

vi.mock("./mode", () => ({
  getAppMode: () => "app",
  modeLabel: () => "Local Backend",
}));

// App and SettingsMenu pull from the same client module; stub the network surface they touch.
vi.mock("../api/client", () => ({
  shutdown: () => shutdown(),
  stopSession: vi.fn(),
  saveFactor: vi.fn(),
  saveWorkspace: vi.fn(),
  listWorkspaces: () => Promise.resolve([]),
  getWorkspace: vi.fn(),
  getSettings: () =>
    Promise.resolve({
      factors_dir: "/d",
      tiingo_api_key_set: false,
      evaluator: "auto",
      cpp_available: false,
    }),
  getDataUsage: () => Promise.resolve([]),
  putSettings: vi.fn(),
  clearData: vi.fn(),
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
  createSession: vi.fn(),
  continueSession: vi.fn(),
  getSession: vi.fn(),
  getSessionRound: vi.fn(),
  getSessionLineage: vi.fn(),
  listSessionRounds: () => Promise.resolve([]),
  listSessions: () => Promise.resolve([]),
}));

import { App } from "./App";

afterEach(() => vi.clearAllMocks());

describe("Quit flow (L7)", () => {
  it("opens the quit dialog and shuts down, then shows the goodbye screen", async () => {
    shutdown.mockResolvedValue({ shutting_down: true });
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Quit" }));
    expect(await screen.findByTestId("quit-dialog")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("quit-confirm"));
    await waitFor(() => expect(shutdown).toHaveBeenCalled());
    expect(await screen.findByTestId("goodbye")).toHaveTextContent(/shut down/i);
  });
});
