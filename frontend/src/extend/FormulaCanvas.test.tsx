import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { FormulaDraftEdge, FormulaDraftNode, PrimitiveInfo } from "../api/types";
import { FormulaCanvas } from "./FormulaCanvas";
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

describe("FormulaCanvas", () => {
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
});
