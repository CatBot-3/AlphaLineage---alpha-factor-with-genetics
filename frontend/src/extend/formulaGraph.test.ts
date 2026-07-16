import { describe, expect, it } from "vitest";
import type { FactorNode, FormulaInputSpec, PrimitiveInfo } from "../api/types";
import {
  autoLayoutGraph,
  blankFormulaGraph,
  bodyToFormulaGraph,
  connectionCreatesCycle,
  deleteFormulaSelection,
  findAvailableNodePosition,
  findAvailableSubgraphPosition,
  formulaNodeHeight,
  graphToFormulaBody,
  primitiveNode,
  removeFormulaInput,
  repairFormulaGraph,
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
    expect(() => graphToFormulaBody(graph.nodes, incomplete)).toThrow(/lookback/i);
  });

  it("folds literal numeric children into inline fields and round-trips them", () => {
    const body: FactorNode = {
      name: "ts_mean",
      children: [{ name: "$arg", value: 0 }, { name: "window", value: 10 }],
    };
    const graph = bodyToFormulaGraph(body, [INPUTS[0]], PRIMITIVES, "series");
    const mean = graph.nodes.find((node) => node.data.primitiveName === "ts_mean")!;
    expect(mean.data.inlineValues).toEqual({ "1": 10 });
    expect(graph.nodes.some((node) => node.data.kind === "value")).toBe(false);
    expect(graph.edges.some((edge) => edge.target === mean.id && edge.targetHandle === "input-1")).toBe(false);
    expect(graphToFormulaBody(graph.nodes, graph.edges)).toEqual(body);
  });

  it("preserves numeric formula-input bindings instead of replacing them with defaults", () => {
    const graph = bodyToFormulaGraph(BODY, INPUTS, PRIMITIVES, "signal");
    const mean = graph.nodes.find((node) => node.data.primitiveName === "ts_mean")!;
    expect(mean.data.inlineValues).toEqual({ "1": null });
    expect(graphToFormulaBody(graph.nodes, graph.edges)).toEqual(BODY);
  });

  it("validates inline windows before serializing", () => {
    const body: FactorNode = {
      name: "ts_mean",
      children: [{ name: "$arg", value: 0 }, { name: "window", value: 10 }],
    };
    const graph = bodyToFormulaGraph(body, [INPUTS[0]], PRIMITIVES, "series");
    const invalid = graph.nodes.map((node) => node.data.primitiveName === "ts_mean"
      ? { ...node, data: { ...node.data, inlineValues: { "1": 1.5 } } }
      : node);
    expect(() => graphToFormulaBody(invalid, graph.edges)).toThrow(/whole number/i);
  });

  it("repairs missing structural nodes in persisted drafts", () => {
    const repaired = repairFormulaGraph([], [], INPUTS, "signal");
    expect(repaired.nodes.map((node) => node.id)).toEqual([
      "formula-input-0",
      "formula-input-1",
      "formula-output",
    ]);
    expect(repaired.nodes.every((node) => node.data.locked)).toBe(true);
  });

  it("folds legacy standalone numeric nodes while repairing a draft", () => {
    const graph = blankFormulaGraph([INPUTS[0]], "series");
    const mean = primitiveNode(PRIMITIVES[0], "mean", 300, 100);
    const windowNode = {
      id: "window-legacy",
      type: "formula",
      x: 100,
      y: 200,
      data: { kind: "value", label: "Window", primitiveName: "window", outType: "window", value: 12, valueKind: "window" },
    };
    const repaired = repairFormulaGraph([...graph.nodes, mean, windowNode], [
      { id: "series-mean", source: "formula-input-0", target: mean.id, sourceHandle: "output", targetHandle: "input-0" },
      { id: "window-mean", source: windowNode.id, target: mean.id, sourceHandle: "output", targetHandle: "input-1" },
      { id: "mean-output", source: mean.id, target: "formula-output", sourceHandle: "output", targetHandle: "result" },
    ], [INPUTS[0]], "series");
    expect(repaired.nodes.some((node) => node.id === windowNode.id)).toBe(false);
    expect(repaired.edges.some((edge) => edge.id === "window-mean")).toBe(false);
    expect(repaired.nodes.find((node) => node.id === mean.id)?.data.inlineValues).toEqual({ "1": 12 });
    expect(graphToFormulaBody(repaired.nodes, repaired.edges)).toEqual({
      name: "ts_mean",
      children: [{ name: "$arg", value: 0 }, { name: "window", value: 12 }],
    });
  });

  it("remaps later input edges when an earlier input is removed", () => {
    const threeInputs = [...INPUTS, { name: "other", type: "series", description: "Other series." }];
    const graph = blankFormulaGraph(threeInputs, "signal");
    const rank = primitiveNode(PRIMITIVES[1], "formula-node-1", 300, 100);
    const nodes = [...graph.nodes, rank];
    const edges = [
      { id: "arg-rank", source: "formula-input-2", target: rank.id, sourceHandle: "output", targetHandle: "input-0" },
      { id: "rank-output", source: rank.id, target: "formula-output", sourceHandle: "output", targetHandle: "result" },
    ];
    const nextInputs = [threeInputs[0], threeInputs[2]];
    const repaired = removeFormulaInput(nodes, edges, 1, nextInputs, "signal");
    expect(repaired.edges).toContainEqual(expect.objectContaining({
      source: "formula-input-1",
      target: rank.id,
      targetHandle: "input-0",
    }));
  });

  it("uses measured node-height estimates for layout and insertion collisions", () => {
    const wide: PrimitiveInfo = {
      name: "wide",
      kind: "operator",
      arg_types: ["series", "series", "series", "series"],
      out_type: "series",
      user: false,
    };
    const first = primitiveNode(wide, "wide-1", 0, 0);
    const second = primitiveNode(wide, "wide-2", 0, 0);
    const layout = autoLayoutGraph([first, second], []);
    expect(Math.abs(layout.nodes[1].y - layout.nodes[0].y)).toBeGreaterThan(formulaNodeHeight(first));
    expect(findAvailableNodePosition([first], { x: 0, y: 0 }, second)).not.toEqual({ x: 0, y: 0 });
  });

  it("moves a pasted subgraph together without changing relative positions", () => {
    const first = primitiveNode(PRIMITIVES[1], "first", 0, 0);
    const second = primitiveNode(PRIMITIVES[1], "second", 240, 100);
    const copies = [
      { ...first, id: "first-copy", x: 35, y: 35 },
      { ...second, id: "second-copy", x: 275, y: 135 },
    ];
    const placed = findAvailableSubgraphPosition([first, second], copies);
    expect(placed[1].x - placed[0].x).toBe(copies[1].x - copies[0].x);
    expect(placed[1].y - placed[0].y).toBe(copies[1].y - copies[0].y);
    expect(placed.map(({ x, y }) => ({ x, y }))).not.toEqual(copies.map(({ x, y }) => ({ x, y })));
  });

  it("deletes selected edges and nodes without deleting structural nodes", () => {
    const graph = blankFormulaGraph([INPUTS[0]], "signal");
    const rank = primitiveNode(PRIMITIVES[1], "rank-node", 300, 100);
    const nodes = [...graph.nodes, rank];
    const edges = [
      { id: "input-rank", source: "formula-input-0", target: rank.id, sourceHandle: "output", targetHandle: "input-0" },
      { id: "rank-output", source: rank.id, target: "formula-output", sourceHandle: "output", targetHandle: "result" },
    ];

    const edgeOnly = deleteFormulaSelection(nodes, edges, ["formula-input-0", "formula-output"], ["input-rank"]);
    expect(edgeOnly.nodes).toHaveLength(nodes.length);
    expect(edgeOnly.edges.map((edge) => edge.id)).toEqual(["rank-output"]);

    const nodeDeletion = deleteFormulaSelection(nodes, edges, [rank.id], []);
    expect(nodeDeletion.nodes.some((node) => node.id === rank.id)).toBe(false);
    expect(nodeDeletion.nodes.some((node) => node.id === "formula-output")).toBe(true);
    expect(nodeDeletion.edges).toEqual([]);
  });
});
