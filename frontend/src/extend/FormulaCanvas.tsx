import {
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  Controls,
  Handle,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type NodeProps,
  Position,
  ReactFlow,
  type ReactFlowInstance,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { DragEvent } from "react";
import type { FormulaDraftEdge, FormulaDraftNode } from "../api/types";
import {
  connectionCreatesCycle,
  nodeOutputType,
  targetInputType,
  typesCompatible,
  type FormulaNodeData,
} from "./formulaGraph";

interface InteractiveData extends FormulaNodeData {
  selectedSlot?: number | null;
  onSelectSlot?: (nodeId: string, index: number) => void;
  onDropPrimitive?: (nodeId: string, index: number, primitiveName: string) => void;
  onValueChange?: (nodeId: string, value: number) => void;
}

type FormulaFlowNode = Node<InteractiveData, "formula">;

function FormulaBlock({ id, data, selected }: NodeProps<FormulaFlowNode>) {
  const inputs = data.inputTypes ?? [];
  return (
    <div className={`formula-block formula-block--${data.kind}${selected ? " is-selected" : ""}`}>
      <div className="formula-block__head">
        <span className="formula-block__source">{data.origin?.replace("_", " ") ?? data.kind}</span>
        <strong>{data.label}</strong>
        {data.revision && data.revision > 1 && <span className="formula-block__revision">v{data.revision}</span>}
      </div>

      {inputs.map((type, index) => (
        <div
          className={`formula-block__input${data.selectedSlot === index ? " is-targeted" : ""}`}
          key={`${id}-input-${index}`}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
          }}
          onDrop={(event) => {
            event.preventDefault();
            event.stopPropagation();
            const primitive = event.dataTransfer.getData("application/x-alphalineage-primitive");
            if (primitive) data.onDropPrimitive?.(id, index, primitive);
          }}
        >
          <Handle
            type="target"
            position={Position.Left}
            id={`input-${index}`}
            style={{ top: 54 + index * 31 }}
          />
          <button
            type="button"
            className="formula-block__socket nodrag"
            title={data.inputDescriptions?.[index]}
            onClick={(event) => {
              event.stopPropagation();
              data.onSelectSlot?.(id, index);
            }}
          >
            <span>{data.inputNames?.[index] ?? `input ${index + 1}`}</span>
            <small>{type}</small>
          </button>
        </div>
      ))}

      {data.kind === "value" && (
        <input
          className="formula-block__value nodrag"
          aria-label={`${data.label} value`}
          type="number"
          step={data.valueKind === "window" ? 1 : "any"}
          value={Number(data.value ?? 0)}
          onChange={(event) => data.onValueChange?.(id, Number(event.target.value))}
        />
      )}

      {data.kind === "factor" && <span className="formula-block__locked">Saved snapshot</span>}
      {data.kind === "output" && <span className="formula-block__locked">Required result</span>}
      <span className="formula-block__type">{data.outType}</span>
      {data.kind === "output" && (
        <Handle type="target" position={Position.Left} id="result" />
      )}
      {data.kind !== "output" && (
        <Handle type="source" position={Position.Right} id="output" />
      )}
    </div>
  );
}

const NODE_TYPES = { formula: FormulaBlock };

function toFlowNode(
  node: FormulaDraftNode,
  callbacks: Omit<InteractiveData, keyof FormulaNodeData>,
  selectedNodeId: string | null,
): FormulaFlowNode {
  const data = node.data as FormulaNodeData;
  return {
    id: node.id,
    type: "formula",
    position: { x: node.x, y: node.y },
    initialWidth: 176,
    initialHeight: Math.max(72, 72 + (data.inputTypes?.length ?? 0) * 31),
    data: { ...data, ...callbacks },
    selected: node.id === selectedNodeId,
  };
}

function fromFlowNode(node: Node): FormulaDraftNode {
  const { onSelectSlot: _select, onDropPrimitive: _drop, onValueChange: _value, selectedSlot: _slot, ...data } = node.data as InteractiveData;
  return { id: node.id, type: "formula", x: node.position.x, y: node.position.y, data };
}

function toFlowEdge(edge: FormulaDraftEdge): Edge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle,
    targetHandle: edge.targetHandle,
    animated: false,
  };
}

function fromFlowEdge(edge: Edge): FormulaDraftEdge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle,
    targetHandle: edge.targetHandle,
  };
}

export function FormulaCanvas({
  nodes,
  edges,
  selectedNodeId,
  selectedSlot,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onSelectNode,
  onSelectSlot,
  onDropPrimitive,
  onDropCanvas,
  onValueChange,
  onInit,
}: {
  nodes: FormulaDraftNode[];
  edges: FormulaDraftEdge[];
  selectedNodeId: string | null;
  selectedSlot: { nodeId: string; index: number } | null;
  onNodesChange: (nodes: FormulaDraftNode[]) => void;
  onEdgesChange: (edges: FormulaDraftEdge[]) => void;
  onConnect: (connection: Connection) => void;
  onSelectNode: (id: string | null) => void;
  onSelectSlot: (nodeId: string, index: number) => void;
  onDropPrimitive: (nodeId: string, index: number, primitiveName: string) => void;
  onDropCanvas: (primitiveName: string, point: { x: number; y: number }) => void;
  onValueChange: (nodeId: string, value: number) => void;
  onInit: (instance: ReactFlowInstance) => void;
}) {
  const callbacks = {
    selectedSlot: null,
    onSelectSlot,
    onDropPrimitive,
    onValueChange,
  };
  const flowNodes = nodes.map((node) => toFlowNode(
    node,
    {
      ...callbacks,
      selectedSlot: selectedSlot?.nodeId === node.id ? selectedSlot.index : null,
    },
    selectedNodeId,
  ));
  const flowEdges = edges.map(toFlowEdge);

  function valid(connection: Connection | Edge): boolean {
    const source = nodes.find((node) => node.id === connection.source);
    const target = nodes.find((node) => node.id === connection.target);
    if (!source || !target || connection.source === connection.target) return false;
    const expected = targetInputType(target, connection.targetHandle);
    if (!expected || !typesCompatible(nodeOutputType(source), expected)) return false;
    if (connectionCreatesCycle(edges, connection.source, connection.target)) return false;
    return !edges.some((edge) => edge.target === connection.target && edge.targetHandle === connection.targetHandle);
  }

  return (
    <div
      className="formula-canvas"
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
    >
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={NODE_TYPES}
        fitView
        minZoom={0.25}
        maxZoom={1.8}
        onInit={(instance) => onInit(instance as unknown as ReactFlowInstance)}
        isValidConnection={valid}
        onConnect={(connection) => {
          if (valid(connection)) onConnect(connection);
        }}
        onNodesChange={(changes: NodeChange[]) => {
          // React Flow owns measured dimensions. Persisting a measurement-only
          // change strips that internal state from our serializable draft and
          // causes controlled nodes to remain hidden as "unmeasured".
          const draftChanges = changes.filter((change) => change.type !== "dimensions");
          if (draftChanges.length > 0) {
            onNodesChange(applyNodeChanges(draftChanges, flowNodes).map(fromFlowNode));
          }
        }}
        onEdgesChange={(changes: EdgeChange[]) => {
          onEdgesChange(applyEdgeChanges(changes, flowEdges).map(fromFlowEdge));
        }}
        onNodeClick={(_event, node) => onSelectNode(node.id)}
        onPaneClick={() => onSelectNode(null)}
        onDrop={(event: DragEvent) => {
          event.preventDefault();
          const primitive = event.dataTransfer.getData("application/x-alphalineage-primitive");
          if (!primitive) return;
          const instance = event.currentTarget.querySelector(".react-flow")?.getBoundingClientRect();
          onDropCanvas(primitive, {
            x: event.clientX - (instance?.left ?? 0),
            y: event.clientY - (instance?.top ?? 0),
          });
        }}
      >
        <Background gap={20} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
