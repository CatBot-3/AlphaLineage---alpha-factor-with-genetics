import { describe, expect, it } from "vitest";
import type { FactorNode, FormulaInputSpec, PrimitiveInfo } from "../api/types";
import {
  bodyToFormulaGraph,
  connectionCreatesCycle,
  graphToFormulaBody,
  targetInputType,
  typesCompatible,
} from "./formulaGraph";

const INPUTS: FormulaInputSpec[] = [
  { name: "price", type: "series", description: "Price input." },
  { name: "lookback", type: "window", description: "Lookback input." },
];

const PRIMITIVES: PrimitiveInfo[] = [
  { name: "ts_mean", display_name: "Moving average", kind: "operator", arg_types: ["series", "window"], inputs: [{ name: "series", type: "series", description: "" }, { name: "lookback", type: "window", description: "" }], out_type: "series", user: false },
  { name: "rank", display_name: "Rank", kind: "operator", arg_types: ["series"], inputs: [{ name: "series", type: "series", description: "" }], out_type: "signal", user: false },
];

const BODY: FactorNode = {
  name: "rank",
  children: [{
    name: "ts_mean",
    children: [{ name: "$arg", value: 0 }, { name: "$arg", value: 1 }],
  }],
};

describe("formula graph", () => {
  it("round-trips by socket identity rather than node position", () => {
    const graph = bodyToFormulaGraph(BODY, INPUTS, PRIMITIVES, "signal");
    const moved = graph.nodes.map((node, index) => ({ ...node, x: 900 - index * 150 }));
    expect(graphToFormulaBody(moved, graph.edges)).toEqual(BODY);
  });

  it("reports expected socket types and panel compatibility", () => {
    const graph = bodyToFormulaGraph(BODY, INPUTS, PRIMITIVES, "signal");
    const mean = graph.nodes.find((node) => node.data.primitiveName === "ts_mean")!;
    expect(targetInputType(mean, "input-1")).toBe("window");
    expect(typesCompatible("signal", "series")).toBe(true);
    expect(typesCompatible("scalar", "window")).toBe(false);
  });

  it("detects a connection that would close a cycle", () => {
    expect(connectionCreatesCycle([
      { id: "a-b", source: "a", target: "b", targetHandle: "input-0" },
      { id: "b-c", source: "b", target: "c", targetHandle: "input-0" },
    ], "c", "a")).toBe(true);
  });

  it("rejects incomplete required sockets", () => {
    const graph = bodyToFormulaGraph(BODY, INPUTS, PRIMITIVES, "signal");
    const incomplete = graph.edges.filter((edge) => edge.source !== "formula-input-1");
    expect(() => graphToFormulaBody(graph.nodes, incomplete)).toThrow(/connect lookback/i);
  });
});
