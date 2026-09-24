import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const { listFormulas, resolveFormulaSource } = vi.hoisted(() => ({
  listFormulas: vi.fn(), resolveFormulaSource: vi.fn(),
}));
vi.mock("../factor/FactorTree", () => ({ FactorTree: () => <div>Formula graph</div> }));

vi.mock("../api/client", () => ({
  listFormulaResults: () =>
    Promise.resolve([
      {
        id: "result-1",
        name: "Momentum result",
        saved_at: "2026-07-14T00:00:00Z",
        tree: { name: "close" },
        metrics: { oos_ic: 0.04 },
        provenance: { universe: "sp500-lite" },
      },
    ]),
  deleteFormulaResult: vi.fn(),
  updateFormulaResult: vi.fn(),
  listFormulas,
  resolveFormulaSource,
}));

import { LibraryPanel } from "./LibraryPanel";

describe("LibraryPanel terminology", () => {
  afterEach(() => vi.clearAllMocks());
  it("presents compatible saved factors as Formula Results", async () => {
    render(<LibraryPanel onSeed={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "Formula Results" })).toBeInTheDocument();
    expect(await screen.findByText("Momentum result")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Seed training from Formula Results (0)" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Saved factors")).not.toBeInTheDocument();
  });

  it("carries the exact saved identity and universe into Signals and training", async () => {
    const onApply = vi.fn(), onSeed = vi.fn();
    render(<LibraryPanel onSeed={onSeed} onApply={onApply}/>);
    await screen.findByText("Momentum result");
    fireEvent.click(screen.getByRole("button", {name: "Apply in Signals"}));
    expect(onApply).toHaveBeenCalledWith("result:result-1", "sp500-lite");
    fireEvent.click(screen.getByRole("button", {name: "Seed training"}));
    expect(onSeed).toHaveBeenCalledWith(["result-1"]);
  });

  it("retries a failed reusable formula load without showing an empty library", async () => {
    listFormulas.mockRejectedValueOnce(new Error("Provider unavailable"))
      .mockResolvedValueOnce([{name: "sma", runtime_name: "ta_sma_r1", display_name: "Moving average", revision: 1}]);
    render(<LibraryPanel onSeed={vi.fn()}/>);
    await screen.findByText("Momentum result");
    fireEvent.click(screen.getByRole("button", {name: "Reusable formulas"}));
    expect(await screen.findByRole("alert")).toHaveTextContent("Provider unavailable");
    expect(screen.queryByText(/No reusable formulas/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {name: "Retry loading"}));
    await waitFor(() => expect(screen.getByText(/Moving average · revision 1/)).toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
