import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Lineage } from "../api/types";
import { LineageDetail } from "./LineageDetail";
import { lineageToFlow } from "./lineageToFlow";

const lineage: Lineage = {
  run_id: "t",
  metadata: {},
  nodes: [
    { id: 0, generation: 0, op: "init", parents: [], tree: { name: "close" } },
    { id: 1, generation: 0, op: "init", parents: [], tree: { name: "volume" } },
    {
      id: 2,
      generation: 1,
      op: "crossover",
      parents: [0, 1],
      tree: { name: "add", children: [{ name: "close" }, { name: "volume" }] },
    },
  ],
};

describe("genealogy (P6-T2)", () => {
  it("connects every parent to its child, carrying the operation", () => {
    const { edges } = lineageToFlow(lineage);
    expect(edges).toHaveLength(2); // 0 -> 2 and 1 -> 2
    expect(edges.every((e) => e.target === "2")).toBe(true);
    expect(edges.map((e) => e.label)).toEqual(["crossover", "crossover"]);
  });

  it("traces parent -> operation -> child and is navigable", () => {
    const onSelect = vi.fn();
    render(<LineageDetail lineage={lineage} selectedId={2} onSelect={onSelect} />);

    expect(screen.getByTestId("lineage-op")).toHaveTextContent("crossover");
    const parents = screen.getAllByTestId("lineage-parent");
    expect(parents).toHaveLength(2);

    fireEvent.click(parents[0]); // clicking a parent navigates to it
    expect(onSelect).toHaveBeenCalledWith(0);
  });
});
