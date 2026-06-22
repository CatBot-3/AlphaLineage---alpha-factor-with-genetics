import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const listUniverses = vi.fn();
const listFormulas = vi.fn();
const getPrimitives = vi.fn();
const getCategories = vi.fn();

vi.mock("../api/client", () => ({
  listUniverses: () => listUniverses(),
  listFormulas: () => listFormulas(),
  getPrimitives: () => getPrimitives(),
  getCategories: () => getCategories(),
}));

import { ExtendPanel } from "./ExtendPanel";

afterEach(() => {
  vi.clearAllMocks();
});

function setupMocks() {
  listUniverses.mockResolvedValue([]);
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

  it("renders Sync Data when page=sync", () => {
    setupMocks();
    render(<ExtendPanel page="sync" />);
    expect(screen.getByTestId("sync-data-page")).toBeInTheDocument();
    expect(screen.queryByTestId("universe-editor-page")).not.toBeInTheDocument();
  });

  it("renders Formula Editor when page=formula", async () => {
    setupMocks();
    render(<ExtendPanel page="formula" />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    expect(screen.getByTestId("formula-editor-page")).toBeInTheDocument();
  });
});
