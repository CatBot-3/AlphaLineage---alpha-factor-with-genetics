import { describe, expect, it } from "vitest";
import { type ComposerEdge, type ComposerNode, findRoot, graphToBody } from "./graphToBody";

const nodes: ComposerNode[] = [
  { id: "s", kind: "sub", x: 50 },
  { id: "a0", kind: "$arg", argIndex: 0, x: 0 },
  { id: "a1", kind: "$arg", argIndex: 1, x: 100 },
];
const edges: ComposerEdge[] = [
  { source: "s", target: "a0" },
  { source: "s", target: "a1" },
];

describe("graphToBody (P7-T2)", () => {
  it("finds the unique output (root) node", () => {
    expect(findRoot(nodes, edges)).toBe("s");
  });

  it("builds a body tree with children ordered left-to-right", () => {
    expect(graphToBody(nodes, edges, "s")).toEqual({
      name: "sub",
      children: [
        { name: "$arg", value: 0 },
        { name: "$arg", value: 1 },
      ],
    });
  });

  it("encodes ephemeral leaf values", () => {
    const single: ComposerNode[] = [{ id: "w", kind: "window", value: 5, x: 0 }];
    expect(graphToBody(single, [], "w")).toEqual({ name: "window", value: 5 });
  });
});
