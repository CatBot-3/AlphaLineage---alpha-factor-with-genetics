import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const listUniverses = vi.fn();
const listFormulas = vi.fn();
const getPrimitives = vi.fn();

vi.mock("../api/client", () => ({
  listUniverses: () => listUniverses(),
  listFormulas: () => listFormulas(),
  getPrimitives: () => getPrimitives(),
}));

import { ExtendPanel } from "./ExtendPanel";

afterEach(() => {
  vi.clearAllMocks();
});

function setupMocks() {
  listUniverses.mockResolvedValue([]);
  listFormulas.mockResolvedValue([]);
  getPrimitives.mockResolvedValue([]);
}

describe("ExtendPanel", () => {
  it("defaults to the Universe Editor page", async () => {
    setupMocks();
    render(<ExtendPanel />);
    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    expect(screen.getByTestId("universe-editor-page")).toBeInTheDocument();
    expect(screen.queryByTestId("sync-data-page")).not.toBeInTheDocument();
    expect(screen.queryByTestId("formula-editor-page")).not.toBeInTheDocument();
  });

  it("switches between all three pages via the dropdown", async () => {
    setupMocks();
    render(<ExtendPanel />);
    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    const select = screen.getByLabelText("Extend page");

    fireEvent.change(select, { target: { value: "sync" } });
    expect(screen.getByTestId("sync-data-page")).toBeInTheDocument();
    expect(screen.queryByTestId("universe-editor-page")).not.toBeInTheDocument();

    fireEvent.change(select, { target: { value: "formula" } });
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    expect(screen.getByTestId("formula-editor-page")).toBeInTheDocument();
    expect(screen.queryByTestId("sync-data-page")).not.toBeInTheDocument();
  });

  it("opens directly on Formula Editor when initialPage is set", async () => {
    setupMocks();
    render(<ExtendPanel initialPage="formula" />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    expect(screen.getByTestId("formula-editor-page")).toBeInTheDocument();
  });
});
