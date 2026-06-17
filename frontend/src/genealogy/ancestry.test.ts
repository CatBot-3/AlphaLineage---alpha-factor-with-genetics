import { describe, expect, it } from "vitest";
import type { Lineage } from "../api/types";
import { ancestorClosure, bestFinalNode } from "./ancestry";

const tree = { name: "close" };

const lineage: Lineage = {
  run_id: "r",
  metadata: {},
  nodes: [
    { id: 0, generation: 0, op: "init", parents: [], tree, fitness: 0.1 },
    { id: 1, generation: 0, op: "seed", parents: [], tree, fitness: 0.5 },
    { id: 2, generation: 1, op: "elite", parents: [1], tree, fitness: 0.5 },
    { id: 3, generation: 1, op: "crossover", parents: [1, 0], tree, fitness: 0.7 },
    { id: 4, generation: 1, op: "subtree_mut", parents: [0], tree, fitness: 0.2 },
  ],
};

describe("ancestry (B4)", () => {
  it("returns exactly the ancestor closure of a node", () => {
    const closure = ancestorClosure(lineage, 3);
    expect(new Set(closure.nodes.map((n) => n.id))).toEqual(new Set([3, 1, 0]));
    // unrelated nodes are excluded
    expect(closure.nodes.some((n) => n.id === 4)).toBe(false);
  });

  it("defaults focus to the fittest final-generation node", () => {
    expect(bestFinalNode(lineage)).toBe(3); // gen 1, fitness 0.7
  });

  it("returns null for an empty lineage", () => {
    expect(bestFinalNode({ run_id: "r", metadata: {}, nodes: [] })).toBeNull();
  });

  it("does not throw on a missing/malformed lineage", () => {
    expect(bestFinalNode(undefined)).toBeNull();
    expect(bestFinalNode({} as unknown as Lineage)).toBeNull();
    expect(ancestorClosure({ run_id: "r", metadata: {}, nodes: [] }, 5).nodes).toEqual([]);
  });
});
