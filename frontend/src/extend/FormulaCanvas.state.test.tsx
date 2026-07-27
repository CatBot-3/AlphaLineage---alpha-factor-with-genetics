import type {
  Edge,
  NodeChange,
  ReactFlowProps,
} from "@xyflow/react";
import { act, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { FormulaDraftEdge, FormulaDraftNode, PrimitiveInfo } from "../api/types";
import { FormulaCanvas, type FormulaFlowNode } from "./FormulaCanvas";
import { blankFormulaGraph, primitiveNode } from "./formulaGraph";

const flowCapture = vi.hoisted(() => ({
  props: null as ReactFlowProps<FormulaFlowNode, Edge> | null,
}));

vi.mock("@xyflow/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@xyflow/react")>();
  return {
    ...actual,
    ReactFlow: (props: ReactFlowProps<FormulaFlowNode, Edge>) => {
      flowCapture.props = props;
      return <div data-testid="react-flow-bridge" />;
    },
    Background: () => null,
  };
});

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

function renderBridge(
  nodes: FormulaDraftNode[],
  edges: FormulaDraftEdge[],
  handlers: {
    onNodesChange: ReturnType<typeof vi.fn>;
    onEdgesChange: ReturnType<typeof vi.fn>;
    onNodeDragStop: ReturnType<typeof vi.fn>;
  },
) {
  return render(
    <FormulaCanvas
      mode="edit"
      nodes={nodes}
      edges={edges}
      selectedNodeId={null}
      selectedSlot={null}
      onNodesChange={handlers.onNodesChange}
      onEdgesChange={handlers.onEdgesChange}
      onConnect={vi.fn()}
      onSelectNode={vi.fn()}
      onSelectSlot={vi.fn()}
      onClearSlot={vi.fn()}
      onDropPrimitive={vi.fn()}
      onDropCanvas={vi.fn()}
      onValueChange={vi.fn()}
      onNodeDragStop={handlers.onNodeDragStop}
      onInit={vi.fn()}
    />,
  );
}

describe("FormulaCanvas React Flow state bridge", () => {
  it("removes every graph mutation hook in inspect mode", () => {
    const node = primitiveNode(MOVING_AVERAGE, "mean", 100, 100);
    render(
      <FormulaCanvas
        mode="inspect"
        nodes={[node]}
        edges={[]}
        selectedNodeId={null}
      />,
    );

    expect(flowCapture.props?.nodesDraggable).toBe(false);
    expect(flowCapture.props?.nodesConnectable).toBe(false);
    expect(flowCapture.props?.onConnect).toBeUndefined();
    expect(flowCapture.props?.onEdgesChange).toBeUndefined();
    expect(flowCapture.props?.onNodeDragStop).toBeUndefined();
  });

  it("keeps transient changes local and commits only final positions on drag-stop", () => {
    const structural = blankFormulaGraph([], "series");
    const first = primitiveNode(MOVING_AVERAGE, "first", 100, 100);
    const second = primitiveNode(MOVING_AVERAGE, "second", 400, 100);
    const nodes = [...structural.nodes, first, second];
    const edges: FormulaDraftEdge[] = [{
      id: "independent-edge",
      source: second.id,
      target: "formula-output",
      sourceHandle: "output",
      targetHandle: "result",
    }];
    const handlers = {
      onNodesChange: vi.fn(),
      onEdgesChange: vi.fn(),
      onNodeDragStop: vi.fn(),
    };
    renderBridge(nodes, edges, handlers);

    act(() => {
      flowCapture.props?.onNodesChange?.([{
        id: first.id,
        type: "dimensions",
        dimensions: { width: 240, height: 150 },
        setAttributes: true,
      }]);
    });
    act(() => {
      flowCapture.props?.onNodesChange?.([
        { id: first.id, type: "select", selected: true },
        {
          id: first.id,
          type: "position",
          position: { x: 250, y: 275 },
          dragging: true,
        },
      ]);
    });

    const canvasFirst = (flowCapture.props?.nodes ?? []).find((node) => node.id === first.id);
    expect(canvasFirst).toMatchObject({
      measured: { width: 240, height: 150 },
      selected: true,
      dragging: true,
      position: { x: 250, y: 275 },
    });
    expect((flowCapture.props?.edges ?? []).map((edge) => edge.id)).toEqual(["independent-edge"]);
    expect(handlers.onNodesChange).not.toHaveBeenCalled();
    expect(handlers.onEdgesChange).not.toHaveBeenCalled();

    const stopped = {
      ...canvasFirst,
      position: { x: 300, y: 325 },
      dragging: false,
    } as FormulaFlowNode;
    act(() => {
      flowCapture.props?.onNodeDragStop?.(
        {} as MouseEvent,
        stopped,
        [stopped],
      );
    });

    expect(handlers.onNodeDragStop).toHaveBeenCalledOnce();
    const committed = handlers.onNodeDragStop.mock.calls[0][0] as FormulaDraftNode[];
    expect(committed.find((node) => node.id === first.id)).toEqual({
      ...first,
      x: 300,
      y: 325,
    });
    expect(committed.find((node) => node.id === second.id)).toEqual(second);
    expect(JSON.stringify(committed)).not.toMatch(
      /measured|selected|dragging|handles|onSelectSlot/,
    );
  });

  it("ignores React Flow structural changes so explicit graph commands retain ownership", () => {
    const node = primitiveNode(MOVING_AVERAGE, "mean", 100, 100);
    const handlers = {
      onNodesChange: vi.fn(),
      onEdgesChange: vi.fn(),
      onNodeDragStop: vi.fn(),
    };
    renderBridge([node], [], handlers);

    act(() => {
      flowCapture.props?.onNodesChange?.([
        { id: node.id, type: "remove" },
        {
          type: "add",
          item: {
            id: "transient-add",
            type: "formula",
            position: { x: 900, y: 900 },
            data: node.data as FormulaFlowNode["data"],
          },
        },
      ] as NodeChange<FormulaFlowNode>[]);
    });

    expect((flowCapture.props?.nodes ?? []).map((item) => item.id)).toEqual([node.id]);
    expect(handlers.onNodesChange).not.toHaveBeenCalled();
  });
});
