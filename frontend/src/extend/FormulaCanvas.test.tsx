import { Position } from "@xyflow/react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { FormulaDraftEdge, FormulaDraftNode, PrimitiveInfo } from "../api/types";
import {
  applyCanvasNodeChanges,
  commitCanvasPositions,
  FormulaCanvas,
  formulaNodePortSignature,
  reconcileCanvasNodes,
  type FormulaFlowNode,
} from "./FormulaCanvas";
import { blankFormulaGraph, primitiveNode } from "./formulaGraph";

const MOVING_AVERAGE: PrimitiveInfo = {
  name: "ts_mean",
  display_name: "Moving average",
  kind: "operator",
  arg_types: ["series", "window"],
  inputs: [
    { name: "series", type: "series", description: "Series input." },
    { name: "lookback", type: "window", description: "Rolling lookback." },
  ],
  out_type: "series",
  user: false,
};

function renderCanvas(nodes: FormulaDraftNode[], edges: FormulaDraftEdge[], onNodesChange = vi.fn()) {
  const rendered = render(
    <FormulaCanvas
      mode="edit"
      nodes={nodes}
      edges={edges}
      selectedNodeId={null}
      selectedSlot={null}
      onNodesChange={onNodesChange}
      onEdgesChange={vi.fn()}
      onConnect={vi.fn()}
      onSelectNode={vi.fn()}
      onSelectSlot={vi.fn()}
      onClearSlot={vi.fn()}
      onDropPrimitive={vi.fn()}
      onDropCanvas={vi.fn()}
      onValueChange={vi.fn()}
      onInit={vi.fn()}
    />,
  );
  return { ...rendered, onNodesChange };
}

function flowNode(
  node: FormulaDraftNode,
  overrides: Partial<FormulaFlowNode> = {},
): FormulaFlowNode {
  return {
    id: node.id,
    type: "formula",
    position: { x: node.x, y: node.y },
    data: node.data as FormulaFlowNode["data"],
    ...overrides,
  };
}

describe("FormulaCanvas", () => {
  it("renders numeric parameters as values rather than controls in inspect mode", () => {
    const mean = primitiveNode(MOVING_AVERAGE, "mean-readonly", 200, 100);
    const { container } = render(
      <FormulaCanvas
        mode="inspect"
        nodes={[mean]}
        edges={[]}
        selectedNodeId={null}
      />,
    );

    expect(screen.queryByRole("spinbutton", { name: /moving average lookback/i })).not.toBeInTheDocument();
    expect(container.querySelector(".formula-block__inline-value")).toHaveTextContent("20");
    expect(screen.getByRole("region", { name: "Read-only formula diagram" })).toHaveAttribute(
      "aria-readonly",
      "true",
    );
  });

  it("renders numeric arguments inline without graph controls", () => {
    const graph = blankFormulaGraph([], "series");
    const mean = primitiveNode(MOVING_AVERAGE, "mean", 200, 100);
    const { container, onNodesChange } = renderCanvas([...graph.nodes, mean], []);

    expect(container.querySelector(".react-flow__controls")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /lookback/i })).not.toBeInTheDocument();
    const lookback = screen.getByRole("spinbutton", { name: /moving average lookback/i });
    expect(lookback).toHaveValue(20);

    fireEvent.change(lookback, { target: { value: "10" } });
    expect(onNodesChange).toHaveBeenCalledWith(expect.arrayContaining([
      expect.objectContaining({
        id: mean.id,
        data: expect.objectContaining({ inlineValues: { "1": 10 } }),
      }),
    ]));
  });

  it("keeps a legacy numeric formula-input binding visible and non-literal", () => {
    const graph = blankFormulaGraph([
      { name: "series", type: "series", description: "Series input." },
      { name: "lookback", type: "window", description: "Lookback input." },
    ], "series");
    const rawMean = primitiveNode(MOVING_AVERAGE, "mean-bound", 200, 100);
    const mean: FormulaDraftNode = {
      ...rawMean,
      data: { ...rawMean.data, inlineValues: { "1": null } },
    };
    const edges: FormulaDraftEdge[] = [{
      id: "lookback-mean",
      source: "formula-input-1",
      target: mean.id,
      sourceHandle: "output",
      targetHandle: "input-1",
    }];
    const { container } = renderCanvas([...graph.nodes, mean], edges);

    expect(screen.getByText("Bound input")).toBeInTheDocument();
    expect(screen.queryByRole("spinbutton", { name: /moving average lookback/i })).not.toBeInTheDocument();
    expect(container.querySelector(`[data-nodeid="${mean.id}"][data-handleid="input-1"]`)).toBeInTheDocument();
  });

  it("keeps measured geometry through dimension, selection, and position changes", () => {
    const graph = blankFormulaGraph([], "series");
    const mean = primitiveNode(MOVING_AVERAGE, "mean", 200, 100);
    const initial = [...graph.nodes, mean].map((node) => flowNode(node));

    const measured = applyCanvasNodeChanges([{
      id: mean.id,
      type: "dimensions",
      dimensions: { width: 244, height: 156 },
      setAttributes: true,
    }], initial);
    const selected = applyCanvasNodeChanges([{
      id: mean.id,
      type: "select",
      selected: true,
    }], measured);
    const moved = applyCanvasNodeChanges([{
      id: mean.id,
      type: "position",
      position: { x: 480, y: 315 },
      dragging: true,
    }], selected);
    const result = moved.find((node) => node.id === mean.id);

    expect(result).toMatchObject({
      measured: { width: 244, height: 156 },
      width: 244,
      height: 156,
      selected: true,
      dragging: true,
      position: { x: 480, y: 315 },
    });
  });

  it("commits a multi-node drag as positions only and leaves independent edges untouched", () => {
    const first = primitiveNode(MOVING_AVERAGE, "first", 100, 100);
    const second = primitiveNode(MOVING_AVERAGE, "second", 400, 100);
    const third = primitiveNode(MOVING_AVERAGE, "third", 700, 100);
    const domainNodes = [first, second, third];
    const edges: FormulaDraftEdge[] = [
      {
        id: "edge-first-second",
        source: first.id,
        target: second.id,
        sourceHandle: "output",
        targetHandle: "input-0",
      },
      {
        id: "edge-independent",
        source: third.id,
        target: "formula-output",
        sourceHandle: "output",
        targetHandle: "result",
      },
    ];
    const edgeSnapshot = structuredClone(edges);
    const transient = domainNodes.map((node) => flowNode(node, {
      measured: { width: 222, height: 111 },
      selected: node.id !== third.id,
      dragging: node.id !== third.id,
      position: node.id === first.id
        ? { x: 180, y: 240 }
        : node.id === second.id
          ? { x: 480, y: 240 }
          : { x: node.x, y: node.y },
    }));

    const persisted = commitCanvasPositions(domainNodes, transient);

    expect(persisted.map(({ id, x, y }) => ({ id, x, y }))).toEqual([
      { id: first.id, x: 180, y: 240 },
      { id: second.id, x: 480, y: 240 },
      { id: third.id, x: 700, y: 100 },
    ]);
    expect(edges).toEqual(edgeSnapshot);
    expect(edges.map((edge) => edge.id)).toEqual(["edge-first-second", "edge-independent"]);
  });

  it("reconciles domain updates by stable id without resetting an active drag or geometry", () => {
    const mean = primitiveNode(MOVING_AVERAGE, "mean", 200, 100);
    const current = flowNode(mean, {
      position: { x: 525, y: 310 },
      dragging: true,
      measured: { width: 260, height: 170 },
      width: 260,
      height: 170,
    });
    const changedDomain = {
      ...mean,
      data: {
        ...mean.data,
        inputTypes: ["series", "window", "scalar"],
        inputNames: ["series", "lookback", "weight"],
      },
    };
    const incoming = flowNode(changedDomain);

    const [result] = reconcileCanvasNodes([current], [incoming]);

    expect(result.position).toEqual({ x: 525, y: 310 });
    expect(result.measured).toEqual({ width: 260, height: 170 });
    expect(result.width).toBe(260);
    expect(result.height).toBe(170);
    expect(result.data.inputTypes).toEqual(["series", "window", "scalar"]);
    expect(formulaNodePortSignature(current.data)).not.toBe(
      formulaNodePortSignature(result.data),
    );
  });

  it("applies undo and redo positions without losing React Flow measurements", () => {
    const mean = primitiveNode(MOVING_AVERAGE, "mean", 200, 100);
    const afterDrag = flowNode({ ...mean, x: 500, y: 320 }, {
      measured: { width: 246, height: 158 },
      width: 246,
      height: 158,
      dragging: false,
    });

    const [afterUndo] = reconcileCanvasNodes([afterDrag], [flowNode(mean)]);
    const [afterRedo] = reconcileCanvasNodes(
      [afterUndo],
      [flowNode({ ...mean, x: 500, y: 320 })],
    );

    expect(afterUndo.position).toEqual({ x: 200, y: 100 });
    expect(afterRedo.position).toEqual({ x: 500, y: 320 });
    expect(afterUndo.measured).toEqual({ width: 246, height: 158 });
    expect(afterRedo.measured).toEqual({ width: 246, height: 158 });
  });

  it("does not let React Flow removal or transient internals enter the draft", () => {
    const mean = primitiveNode(MOVING_AVERAGE, "mean", 200, 100);
    const canvasNode = flowNode(mean, {
      position: { x: 321, y: 654 },
      measured: { width: 244, height: 156 },
      width: 244,
      height: 156,
      selected: true,
      dragging: true,
      handles: [{
        id: "output",
        type: "source",
        position: Position.Right,
        x: 240,
        y: 70,
        width: 8,
        height: 8,
      }],
      data: {
        ...(mean.data as FormulaFlowNode["data"]),
        onSelectSlot: vi.fn(),
      },
    });

    expect(applyCanvasNodeChanges([{
      id: mean.id,
      type: "remove",
    }], [canvasNode])).toHaveLength(1);

    const [persisted] = commitCanvasPositions([mean], [canvasNode]);
    const serialized = JSON.stringify(persisted);

    expect(persisted.data).toEqual(mean.data);
    expect(serialized).not.toContain("measured");
    expect(serialized).not.toContain("selected");
    expect(serialized).not.toContain("dragging");
    expect(serialized).not.toContain("handles");
    expect(serialized).not.toContain("onSelectSlot");
  });
});
