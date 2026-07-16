import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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
}));

import { LibraryPanel } from "./LibraryPanel";

describe("LibraryPanel terminology", () => {
  it("presents compatible saved factors as Formula Results", async () => {
    render(<LibraryPanel onSeed={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "Formula Results" })).toBeInTheDocument();
    expect(await screen.findByText("Momentum result")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Seed training from Formula Results (0)" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Saved factors")).not.toBeInTheDocument();
  });
});
