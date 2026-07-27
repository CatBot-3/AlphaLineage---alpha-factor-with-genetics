import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const listUniverses = vi.fn();
const listUniversePresets = vi.fn();
const listFormulas = vi.fn();
const getPrimitives = vi.fn();
const getCategories = vi.fn();

vi.mock("../api/client", () => ({
  listUniverses: () => listUniverses(),
  listUniversePresets: () => listUniversePresets(),
  listFormulas: () => listFormulas(),
  listFormulaResults: () => Promise.resolve([]),
  getPrimitives: () => getPrimitives(),
  getCategories: () => getCategories(),
}));

import { ExtendPanel } from "./ExtendPanel";

afterEach(() => {
  vi.clearAllMocks();
});

function setupMocks() {
  listUniverses.mockResolvedValue([]);
  listUniversePresets.mockResolvedValue([]);
  listFormulas.mockResolvedValue([]);
  getPrimitives.mockResolvedValue([]);
  getCategories.mockResolvedValue({ order: [], overrides: {} });
}

describe("ExtendPanel (controlled by the nav dropdown)", () => {
  it("renders the page named by the `page` prop", async () => {
    setupMocks();
    render(<ExtendPanel page="universe" />);
    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    expect(screen.getByTestId("universe-editor-page")).toBeInTheDocument();
    expect(screen.queryByTestId("sync-data-page")).not.toBeInTheDocument();
  });

  it("migrates the legacy page=sync value to Universe Editor", async () => {
    setupMocks();
    render(<ExtendPanel page="sync" />);
    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    expect(screen.getByTestId("universe-editor-page")).toBeInTheDocument();
    expect(screen.getByTestId("universe-data-sync")).toBeInTheDocument();
  });

  it("renders the unified Formula Builder when page=formula", async () => {
    setupMocks();
    render(<ExtendPanel page="formula" />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    expect(screen.getByTestId("formula-editor-page")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Formula Builder" })).toBeInTheDocument();
  });

  it("lets a user open a recovered legacy builder draft without discarding it", async () => {
    setupMocks();
    const onRecoverFormulaDraft = vi.fn();
    render(
      <ExtendPanel
        page="formula"
        recoveredFormulaDrafts={[
          {
            label: "Legacy Factor Builder draft",
            draft: { name: "legacy", display_name: "Legacy", description: "" },
          },
        ]}
        onRecoverFormulaDraft={onRecoverFormulaDraft}
      />,
    );
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());

    fireEvent.click(screen.getByText("Recovered Formula Builder drafts (1)"));
    fireEvent.click(screen.getByRole("button", { name: "Open Legacy Factor Builder draft" }));
    expect(onRecoverFormulaDraft).toHaveBeenCalledWith(0);
  });
});
