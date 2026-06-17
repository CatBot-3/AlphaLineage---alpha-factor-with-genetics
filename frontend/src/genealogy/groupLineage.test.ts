import { describe, expect, it } from "vitest";
import type { Lineage } from "../api/types";
import { groupLineage, primaryMethod } from "./groupLineage";

const tree = { name: "close" };

const lineage: Lineage = {
  run_id: "r",
  metadata: {},
  nodes: [
    { id: 0, generation: 0, op: "init", parents: [], tree, fitness: 0.1 },
    { id: 1, generation: 0, op: "seed", parents: [], tree, fitness: 0.5 },
    { id: 2, generation: 1, op: "elite", parents: [1], tree, fitness: 0.5 },
    { id: 3, generation: 1, op: "crossover+point_mut", parents: [1, 0], tree, fitness: 0.3 },
    { id: 4, generation: 1, op: "subtree_mut", parents: [0], tree, fitness: 0.45 },
    { id: 5, generation: 1, op: "crossover", parents: [2, 1], tree, fitness: 0.38 },
  ],
};

describe("groupLineage (B4)", () => {
  it("groups composite ops under their primary method", () => {
    expect(primaryMethod("crossover+point_mut")).toBe("crossover");
    expect(primaryMethod("reproduction+point_mut")).toBe("point mutation");
    expect(primaryMethod("subtree_mut")).toBe("subtree mutation");
    expect(primaryMethod("elite")).toBe("elite");
  });

  it("puts the latest generation first and groups by method", () => {
    const generations = groupLineage(lineage);
    expect(generations[0].generation).toBe(1);
    const methods = generations[0].groups.map((g) => g.method);
    expect(methods).toContain("crossover");
    expect(methods).toContain("subtree mutation");
    expect(methods).toContain("elite");
    // crossover group folds #3 and #5 together
    const crossover = generations[0].groups.find((g) => g.method === "crossover");
    expect(crossover?.count).toBe(2);
  });

  it("sorts members by fitness descending and groups by best fitness", () => {
    const generations = groupLineage(lineage);
    const crossover = generations[0].groups.find((g) => g.method === "crossover");
    expect(crossover?.members.map((m) => m.id)).toEqual([5, 3]); // 0.38 before 0.30
    // most-promising method first: elite (0.5) leads
    expect(generations[0].groups[0].method).toBe("elite");
    expect(generations[0].bestFitness).toBe(0.5);
  });

  it("tolerates missing fitness (old demo runs)", () => {
    const noFit: Lineage = {
      run_id: "r",
      metadata: {},
      nodes: [{ id: 0, generation: 0, op: "init", parents: [], tree }],
    };
    const generations = groupLineage(noFit);
    expect(generations[0].bestFitness).toBeNull();
    expect(generations[0].groups[0].members[0].fitness).toBeNull();
  });
});
