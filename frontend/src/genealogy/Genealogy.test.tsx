import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Lineage } from "../api/types";
import { Genealogy } from "./Genealogy";

const tree = { name: "close" };

const lineage: Lineage = {
  run_id: "r",
  metadata: {},
  nodes: [
    { id: 0, generation: 0, op: "init", parents: [], tree, fitness: 0.1 },
    { id: 1, generation: 1, op: "crossover", parents: [0], tree, fitness: 0.3 },
    { id: 2, generation: 1, op: "elite", parents: [0], tree, fitness: 0.5 },
  ],
};

describe("Genealogy (B4)", () => {
  it("collapses generations and groups by default", () => {
    render(<Genealogy lineage={lineage} />);
    expect(screen.getByTestId("generation-list")).toBeInTheDocument();
    // nothing expanded -> no method groups or member rows in the DOM
    expect(screen.queryByTestId("method-group")).not.toBeInTheDocument();
    expect(screen.queryByTestId("member-row")).not.toBeInTheDocument();
  });

  it("expands a generation then a group to reveal fitness-sorted members", () => {
    render(<Genealogy lineage={lineage} />);
    fireEvent.click(screen.getByText("Generation 1"));
    // the most-promising method (elite, 0.5) leads
    const groups = screen.getAllByTestId("method-group");
    fireEvent.click(within(groups[0]).getByText("elite"));
    expect(screen.getAllByTestId("member-row").length).toBeGreaterThan(0);
  });

  it("switches to the focused ancestry trace", () => {
    render(<Genealogy lineage={lineage} />);
    fireEvent.click(screen.getByTestId("mode-ancestry"));
    expect(screen.getByTestId("ancestry-view")).toBeInTheDocument();
    expect(screen.getByTestId("ancestry-view")).toHaveTextContent(/ancestor/i);
  });
});
