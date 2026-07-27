import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Lineage } from "../api/types";
import { AncestryView, lineageKeyboardNeighbor } from "./AncestryView";

const lineage: Lineage = {
  run_id: "r",
  metadata: {},
  nodes: [
    { id: 0, generation: 0, op: "init", parents: [], tree: { name: "close" }, fitness: 0.1 },
    { id: 1, generation: 0, op: "init", parents: [], tree: { name: "returns" }, fitness: 0.2 },
    {
      id: 3,
      generation: 1,
      op: "crossover",
      parents: [1, 0],
      tree: { name: "add", children: [{ name: "close" }, { name: "returns" }] },
      fitness: 0.4,
    },
  ],
};

describe("AncestryView keyboard interaction", () => {
  it("moves focus without inspecting until Enter or Space and copies the focused expression", async () => {
    const onSelect = vi.fn();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    render(<AncestryView lineage={lineage} focusId={3} onSelect={onSelect} />);

    const graph = screen.getByRole("region", { name: "Ancestry graph for formula node 3" });
    fireEvent.keyDown(graph, { key: "ArrowUp" });
    expect(onSelect).not.toHaveBeenCalled();

    fireEvent.keyDown(graph, { key: "Enter" });
    expect(onSelect).toHaveBeenLastCalledWith(1);

    fireEvent.keyDown(graph, { key: "c", ctrlKey: true });
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("returns"));
    expect(screen.getByRole("status")).toHaveTextContent("Copied formula 1.");

    fireEvent.keyDown(graph, { key: "ArrowDown" });
    fireEvent.keyDown(graph, { key: " " });
    expect(onSelect).toHaveBeenLastCalledWith(3);
  });

  it("navigates laterally only among formulas in the same generation", () => {
    expect(lineageKeyboardNeighbor(lineage, 0, "right")).toBe(1);
    expect(lineageKeyboardNeighbor(lineage, 1, "left")).toBe(0);
    expect(lineageKeyboardNeighbor(lineage, 0, "left")).toBe(0);
  });
});
