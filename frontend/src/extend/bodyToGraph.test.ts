import { describe, expect, it } from "vitest";
import type { FactorNode } from "../api/types";
import { bodyToGraph } from "./bodyToGraph";
import { findRoot, graphToBody } from "./graphToBody";

const TREES: FactorNode[] = [
  { name: "close" },
  {
    name: "sub",
    children: [
      { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "window", value: 12 }] },
      { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "window", value: 26 }] },
    ],
  },
  {
    name: "div",
    children: [
      { name: "sub", children: [{ name: "$arg", value: 0 }, { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "window", value: 20 }] }] },
      { name: "ts_std", children: [{ name: "$arg", value: 0 }, { name: "window", value: 20 }] },
    ],
  },
];

describe("bodyToGraph", () => {
  it("is the exact inverse of graphToBody (tree -> graph -> tree)", () => {
    for (const tree of TREES) {
      const { nodes, edges } = bodyToGraph(tree);
      const root = findRoot(nodes, edges);
      expect(root).not.toBeNull();
      expect(graphToBody(nodes, edges, root as string)).toEqual(tree);
    }
  });

  it("assigns strictly increasing x in argument order", () => {
    const { nodes, edges } = bodyToGraph(TREES[1]);
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const root = findRoot(nodes, edges) as string;
    const childX = edges.filter((e) => e.source === root).map((e) => byId.get(e.target)!.x);
    expect(childX).toEqual([...childX].sort((a, b) => a - b));
  });
});
