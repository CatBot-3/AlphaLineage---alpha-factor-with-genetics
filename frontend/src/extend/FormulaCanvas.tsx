import {
  applyEdgeChanges,
  applyNodeChanges,
  Background,
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
  useUpdateNodeInternals,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useRef, type DragEvent } from "react";
import type { FormulaDraftEdge, FormulaDraftNode } from "../api/types";
import {
  connectionCreatesCycle,
  formulaNodeHeight,
  isInlineInputType,
  nodeOutputType,
  targetInputType,
  typesCompatible,
  type FormulaNodeData,
} from "./formulaGraph";

interface InteractiveData extends FormulaNodeData {
  selectedSlot?: number | null;
  boundInlineInputs?: number[];
  onSelectSlot?: (nodeId: string, index: number) => void;
  onDropPrimitive?: (nodeId: string, index: number, primitiveName: string) => void;
  onValueChange?: (nodeId: string, value: number) => void;
  onInlineValueChange?: (nodeId: string, index: number, value: number | null) => void;
  onInlineValueCommit?: () => void;
}

type FormulaFlowNode = Node<InteractiveData, "formula">;

export interface FormulaCanvasDropPlacement {
  exact: true;
  anchor: "center";
}

function FormulaBlock({ id, data, selected }: NodeProps<FormulaFlowNode>) {
  const inputs = data.inputTypes ?? [];
  const boundInlineInputs = data.boundInlineInputs ?? [];
  const updateNodeInternals = useUpdateNodeInternals();
  const layoutSignature = `${inputs.join("|")}:${boundInlineInputs.join("|")}:${data.kind}`;
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, layoutSignature, updateNodeInternals]);

  return (
    <div className={`formula-block formula-block--${data.kind}${selected ? " is-selected" : ""}`}>
      <div className="formula-block__head">
        <span className="formula-block__source">{data.origin?.replace("_", " ") ?? data.kind}</span>
        <strong>{data.label}</strong>
        {data.revision != null && <span className="formula-block__revision">v{data.revision}</span>}
      </div>

      {inputs.map((type, index) => {
        const inputName = data.inputNames?.[index] ?? `input ${index + 1}`;
        if (isInlineInputType(type)) {
          const bound = boundInlineInputs.includes(index);
          const value = data.inlineValues?.[String(index)];
          return (
            <label
              className={`formula-block__inline nodrag${bound ? " is-bound" : ""}`}
              title={data.inputDescriptions?.[index]}
              key={`${id}-input-${index}`}
            >
              {bound && <Handle type="target" position={Position.Left} id={`input-${index}`} />}
              <span><span>{inputName}</span><small>{type}</small></span>
              {bound ? (
                <span className="formula-block__binding">Bound input</span>
              ) : (
                <input
                  aria-label={`${data.label} ${inputName}`}
                  type="number"
                  step={type === "window" ? 1 : "any"}
                  min={type === "window" ? 1 : undefined}
                  max={type === "window" ? 100000 : undefined}
                  value={value ?? ""}
                  onChange={(event) => {
                    const raw = event.target.value;
                    data.onInlineValueChange?.(id, index, raw === "" ? null : Number(raw));
                  }}
                  onBlur={() => data.onInlineValueCommit?.()}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") event.currentTarget.blur();
                  }}
                />
              )}
            </label>
          );
        }
        return (
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
            <Handle type="target" position={Position.Left} id={`input-${index}`} />
            <button
              type="button"
              className="formula-block__socket nodrag"
              title={data.inputDescriptions?.[index]}
              onClick={(event) => {
                event.stopPropagation();
                data.onSelectSlot?.(id, index);
              }}
            >
              <span>{inputName}</span>
              <small>{type}</small>
            </button>
          </div>
        );
      })}

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
  selectedNodeIds?: ReadonlySet<string>,
): FormulaFlowNode {
  const data = node.data as FormulaNodeData;
  return {
    id: node.id,
    type: "formula",
    position: { x: node.x, y: node.y },
    initialWidth: 176,
    initialHeight: formulaNodeHeight(node),
    data: { ...data, ...callbacks },
    selected: selectedNodeIds ? selectedNodeIds.has(node.id) : node.id === selectedNodeId,
    deletable: data.kind !== "input" && data.kind !== "output",
    draggable: data.kind !== "input",
  };
}

function fromFlowNode(node: Node): FormulaDraftNode {
  const {
    onSelectSlot: _select,
    onDropPrimitive: _drop,
    onValueChange: _value,
    onInlineValueChange: _inlineValue,
    onInlineValueCommit: _inlineCommit,
    selectedSlot: _slot,
    boundInlineInputs: _boundInlineInputs,
    ...data
  } = node.data as InteractiveData;
  return { id: node.id, type: "formula", x: node.position.x, y: node.position.y, data };
}

function toFlowEdge(edge: FormulaDraftEdge, selectedEdgeIds?: ReadonlySet<string>): Edge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle,
    targetHandle: edge.targetHandle,
    animated: false,
    selected: selectedEdgeIds?.has(edge.id) ?? false,
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
  selectedNodeIds,
  selectedEdgeIds,
  selectedSlot,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onSelectNode,
  onSelectionChange,
  onNodeDragStart,
  onNodeDragStop,
  onSelectSlot,
  onClearSlot,
  onDropPrimitive,
  onDropCanvas,
  onValueChange,
  onInlineValueChange,
  onInlineValueCommit,
  onInit,
}: {
  nodes: FormulaDraftNode[];
  edges: FormulaDraftEdge[];
  selectedNodeId: string | null;
  selectedNodeIds?: string[];
  selectedEdgeIds?: string[];
  selectedSlot: { nodeId: string; index: number } | null;
  onNodesChange: (nodes: FormulaDraftNode[]) => void;
  onEdgesChange: (edges: FormulaDraftEdge[]) => void;
  onConnect: (connection: Connection) => void;
  onSelectNode: (id: string | null) => void;
  onSelectionChange?: (nodeIds: string[], edgeIds: string[]) => void;
  onNodeDragStart?: () => void;
  onNodeDragStop?: (nodes: FormulaDraftNode[]) => void;
  onSelectSlot: (nodeId: string, index: number) => void;
  onClearSlot: () => void;
  onDropPrimitive: (nodeId: string, index: number, primitiveName: string) => void;
  onDropCanvas: (
    primitiveName: string,
    point: { x: number; y: number },
    placement?: FormulaCanvasDropPlacement,
  ) => void;
  onValueChange: (nodeId: string, value: number) => void;
  onInlineValueChange?: (nodeId: string, index: number, value: number | null) => void;
  onInlineValueCommit?: () => void;
  onInit: (instance: ReactFlowInstance) => void;
}) {
  const flowInstance = useRef<ReactFlowInstance | null>(null);
  const selectedIdSet = selectedNodeIds ? new Set(selectedNodeIds) : undefined;
  const selectedEdgeIdSet = selectedEdgeIds ? new Set(selectedEdgeIds) : undefined;
  const changeInlineValue = (nodeId: string, index: number, value: number | null) => {
    if (onInlineValueChange) {
      onInlineValueChange(nodeId, index, value);
      return;
    }
    onNodesChange(nodes.map((node) => node.id === nodeId ? {
      ...node,
      data: {
        ...node.data,
        inlineValues: {
          ...((node.data as FormulaNodeData).inlineValues ?? {}),
          [String(index)]: value,
        },
      },
    } : node));
  };
  const callbacks = {
    selectedSlot: null,
    onSelectSlot,
    onDropPrimitive,
    onValueChange,
    onInlineValueChange: changeInlineValue,
    onInlineValueCommit,
  };
  const boundInlineByNode = new Map<string, number[]>();
  for (const edge of edges) {
    if (!edge.targetHandle?.startsWith("input-")) continue;
    const index = Number(edge.targetHandle.slice("input-".length));
    const target = nodes.find((node) => node.id === edge.target);
    const type = target ? (target.data as FormulaNodeData).inputTypes?.[index] : undefined;
    if (!type || !isInlineInputType(type)) continue;
    const indexes = boundInlineByNode.get(edge.target) ?? [];
    indexes.push(index);
    boundInlineByNode.set(edge.target, indexes);
  }
  const flowNodes = nodes.map((node) => toFlowNode(
    node,
    {
      ...callbacks,
      selectedSlot: selectedSlot?.nodeId === node.id ? selectedSlot.index : null,
      boundInlineInputs: boundInlineByNode.get(node.id) ?? [],
    },
    selectedNodeId,
    selectedIdSet,
  ));
  const flowEdges = edges.map((edge) => toFlowEdge(edge, selectedEdgeIdSet));
  const handleSelectionChange = useCallback(({ nodes: selectedNodes, edges: selectedEdges }: {
    nodes: Node[];
    edges: Edge[];
  }) => {
    onSelectionChange?.(
      selectedNodes.map((node) => node.id),
      selectedEdges.map((edge) => edge.id),
    );
  }, [onSelectionChange]);

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
        zoomOnScroll
        zoomOnPinch
        panOnDrag
        deleteKeyCode={null}
        onInit={(instance) => {
          const typed = instance as unknown as ReactFlowInstance;
          flowInstance.current = typed;
          onInit(typed);
        }}
        isValidConnection={valid}
        onConnect={(connection) => {
          if (valid(connection)) onConnect(connection);
        }}
        onNodesChange={(changes: NodeChange[]) => {
          // React Flow owns measured dimensions. Persisting a measurement-only
          // change strips that internal state from our serializable draft and
          // causes controlled nodes to remain hidden as "unmeasured".
          const structuralIds = new Set(flowNodes
            .filter((node) => node.data.kind === "input" || node.data.kind === "output")
            .map((node) => node.id));
          const draftChanges = changes.filter((change) => (
            change.type !== "dimensions" &&
            !(change.type === "remove" && structuralIds.has(change.id))
          ));
          if (draftChanges.length > 0) {
            onNodesChange(applyNodeChanges(draftChanges, flowNodes).map(fromFlowNode));
          }
        }}
        onEdgesChange={(changes: EdgeChange[]) => {
          const persistedChanges = changes.filter((change) => change.type !== "select");
          if (persistedChanges.length > 0) {
            onEdgesChange(applyEdgeChanges(persistedChanges, flowEdges).map(fromFlowEdge));
          }
        }}
        onNodeClick={(_event, node) => {
          onClearSlot();
          onSelectNode(node.id);
        }}
        onNodeDragStart={() => onNodeDragStart?.()}
        onNodeDragStop={(_event, stoppedNode, draggedNodes) => {
          if (!onNodeDragStop) return;
          const moved = new Map(draggedNodes.map((node) => [node.id, node]));
          moved.set(stoppedNode.id, stoppedNode);
          onNodeDragStop(flowNodes.map((node) => fromFlowNode(moved.get(node.id) ?? node)));
        }}
        onSelectionChange={onSelectionChange ? handleSelectionChange : undefined}
        onPaneClick={() => {
          onClearSlot();
          onSelectNode(null);
        }}
        onDrop={(event: DragEvent) => {
          event.preventDefault();
          const primitive = event.dataTransfer.getData("application/x-alphalineage-primitive");
          if (!primitive) return;
          const point = flowInstance.current?.screenToFlowPosition({
            x: event.clientX,
            y: event.clientY,
          });
          if (point) onDropCanvas(primitive, point, { exact: true, anchor: "center" });
        }}
      >
        <Background gap={20} size={1} />
      </ReactFlow>
    </div>
  );
}
