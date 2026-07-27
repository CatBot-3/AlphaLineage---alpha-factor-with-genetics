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
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
} from "react";
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
  readOnly?: boolean;
  selectedSlot?: number | null;
  boundInlineInputs?: number[];
  onSelectSlot?: (nodeId: string, index: number) => void;
  onDropPrimitive?: (nodeId: string, index: number, primitiveName: string) => void;
  onValueChange?: (nodeId: string, value: number) => void;
  onInlineValueChange?: (nodeId: string, index: number, value: number | null) => void;
  onInlineValueCommit?: () => void;
}

export type FormulaFlowNode = Node<InteractiveData, "formula">;

export interface FormulaCanvasDropPlacement {
  exact: true;
  anchor: "center";
}

function FormulaBlock({ id, data, selected }: NodeProps<FormulaFlowNode>) {
  const inputs = data.inputTypes ?? [];
  const boundInlineInputs = data.boundInlineInputs ?? [];
  const readOnly = data.readOnly === true;
  const updateNodeInternals = useUpdateNodeInternals();
  const layoutSignature = formulaNodePortSignature(data);
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
            <div
              className={`formula-block__inline nodrag${bound ? " is-bound" : ""}`}
              title={data.inputDescriptions?.[index]}
              key={`${id}-input-${index}`}
            >
              {bound && (
                <Handle
                  type="target"
                  position={Position.Left}
                  id={`input-${index}`}
                  isConnectable={!readOnly}
                />
              )}
              <span><span>{inputName}</span><small>{type}</small></span>
              {bound ? (
                <span className="formula-block__binding">Bound input</span>
              ) : readOnly ? (
                <output
                  className="formula-block__inline-value"
                  aria-label={`${data.label} ${inputName}`}
                >
                  {value ?? "—"}
                </output>
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
            </div>
          );
        }
        return (
          <div
            className={`formula-block__input${data.selectedSlot === index ? " is-targeted" : ""}`}
            key={`${id}-input-${index}`}
            onDragOver={readOnly ? undefined : (event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
            }}
            onDrop={readOnly ? undefined : (event) => {
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
              isConnectable={!readOnly}
            />
            {readOnly ? (
              <span className="formula-block__socket nodrag" title={data.inputDescriptions?.[index]}>
                <span>{inputName}</span>
                <small>{type}</small>
              </span>
            ) : (
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
            )}
          </div>
        );
      })}

      {data.kind === "value" && readOnly && (
        <output className="formula-block__value formula-block__value--readonly">
          {Number(data.value ?? 0)}
        </output>
      )}
      {data.kind === "value" && !readOnly && (
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
        <Handle
          type="target"
          position={Position.Left}
          id="result"
          isConnectable={!readOnly}
        />
      )}
      {data.kind !== "output" && (
        <Handle
          type="source"
          position={Position.Right}
          id="output"
          isConnectable={!readOnly}
        />
      )}
    </div>
  );
}

const NODE_TYPES = { formula: FormulaBlock };

export function formulaNodePortSignature(data: Pick<
  InteractiveData,
  "inputTypes" | "boundInlineInputs" | "kind"
>): string {
  return `${(data.inputTypes ?? []).join("|")}:${(data.boundInlineInputs ?? []).join("|")}:${data.kind}`;
}

function toFlowNode(
  node: FormulaDraftNode,
  callbacks: Omit<InteractiveData, keyof FormulaNodeData>,
  selectedNodeId: string | null,
  editable: boolean,
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
    deletable: editable && data.kind !== "input" && data.kind !== "output",
    draggable: editable && data.kind !== "input",
  };
}

/**
 * Reconcile persisted formula nodes into React Flow without throwing away
 * measurements, handle bounds, or an in-progress drag. Draft nodes deliberately
 * never receive those React Flow-only fields.
 */
export function reconcileCanvasNodes(
  current: FormulaFlowNode[],
  incoming: FormulaFlowNode[],
): FormulaFlowNode[] {
  const currentById = new Map(current.map((node) => [node.id, node]));
  return incoming.map((node) => {
    const previous = currentById.get(node.id);
    if (!previous) return node;
    return {
      ...previous,
      ...node,
      position: previous.dragging ? previous.position : node.position,
    };
  });
}

/**
 * React Flow owns position, selection, drag and measurement changes locally.
 * Structural removal is routed through the editor's explicit delete command.
 */
export function applyCanvasNodeChanges(
  changes: NodeChange<FormulaFlowNode>[],
  current: FormulaFlowNode[],
): FormulaFlowNode[] {
  return applyNodeChanges(
    changes.filter((change) => (
      change.type === "dimensions" ||
      change.type === "position" ||
      change.type === "select"
    )),
    current,
  );
}

/**
 * Copy only final coordinates back to the serializable domain nodes. Starting
 * from the domain objects guarantees that measurements, selection, callbacks,
 * and other React Flow internals can never leak into workspace persistence.
 */
export function commitCanvasPositions(
  domainNodes: FormulaDraftNode[],
  canvasNodes: ReadonlyArray<Pick<FormulaFlowNode, "id" | "position">>,
): FormulaDraftNode[] {
  const canvasById = new Map(canvasNodes.map((node) => [node.id, node.position]));
  return domainNodes.map((node) => {
    const position = canvasById.get(node.id);
    return position ? { ...node, x: position.x, y: position.y } : node;
  });
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

interface FormulaCanvasBaseProps {
  nodes: FormulaDraftNode[];
  edges: FormulaDraftEdge[];
  selectedNodeId: string | null;
  selectedNodeIds?: string[];
  selectedEdgeIds?: string[];
  onSelectNode?: (id: string | null) => void;
  onSelectionChange?: (nodeIds: string[], edgeIds: string[]) => void;
  onInit?: (instance: ReactFlowInstance) => void;
  ariaLabel?: string;
  compact?: boolean;
}

export interface FormulaCanvasEditProps extends FormulaCanvasBaseProps {
  mode: "edit";
  selectedSlot: { nodeId: string; index: number } | null;
  onNodesChange: (nodes: FormulaDraftNode[]) => void;
  onEdgesChange: (edges: FormulaDraftEdge[]) => void;
  onConnect: (connection: Connection) => void;
  onSelectNode: (id: string | null) => void;
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
}

export interface FormulaCanvasInspectProps extends FormulaCanvasBaseProps {
  mode: "inspect";
}

export type FormulaCanvasProps = FormulaCanvasEditProps | FormulaCanvasInspectProps;

function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

export function FormulaCanvas(props: FormulaCanvasProps) {
  const {
    nodes,
    edges,
    selectedNodeId,
    selectedNodeIds,
    selectedEdgeIds,
    onSelectNode,
    onSelectionChange,
    onInit,
  } = props;
  const editable = props.mode === "edit";
  const selectedSlot = editable ? props.selectedSlot : null;
  const onNodesChange = editable ? props.onNodesChange : undefined;
  const onEdgesChange = editable ? props.onEdgesChange : undefined;
  const onConnect = editable ? props.onConnect : undefined;
  const onNodeDragStart = editable ? props.onNodeDragStart : undefined;
  const onNodeDragStop = editable ? props.onNodeDragStop : undefined;
  const onSelectSlot = editable ? props.onSelectSlot : undefined;
  const onClearSlot = editable ? props.onClearSlot : undefined;
  const onDropPrimitive = editable ? props.onDropPrimitive : undefined;
  const onDropCanvas = editable ? props.onDropCanvas : undefined;
  const onValueChange = editable ? props.onValueChange : undefined;
  const onInlineValueChange = editable ? props.onInlineValueChange : undefined;
  const onInlineValueCommit = editable ? props.onInlineValueCommit : undefined;
  const flowInstance = useRef<ReactFlowInstance | null>(null);
  const canvasNodesRef = useRef<FormulaFlowNode[]>([]);
  const selectedIdSet = useMemo(
    () => selectedNodeIds ? new Set(selectedNodeIds) : undefined,
    [selectedNodeIds],
  );
  const selectedEdgeIdSet = useMemo(
    () => selectedEdgeIds ? new Set(selectedEdgeIds) : undefined,
    [selectedEdgeIds],
  );
  const changeInlineValue = useCallback((nodeId: string, index: number, value: number | null) => {
    if (!onNodesChange) return;
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
  }, [nodes, onInlineValueChange, onNodesChange]);
  const callbacks = useMemo(() => ({
    readOnly: !editable,
    selectedSlot: null,
    onSelectSlot,
    onDropPrimitive,
    onValueChange,
    onInlineValueChange: changeInlineValue,
    onInlineValueCommit,
  }), [
    changeInlineValue,
    editable,
    onDropPrimitive,
    onInlineValueCommit,
    onSelectSlot,
    onValueChange,
  ]);
  const boundInlineByNode = useMemo(() => {
    const result = new Map<string, number[]>();
    const nodesById = new Map(nodes.map((node) => [node.id, node]));
    for (const edge of edges) {
      if (!edge.targetHandle?.startsWith("input-")) continue;
      const index = Number(edge.targetHandle.slice("input-".length));
      const target = nodesById.get(edge.target);
      const type = target ? (target.data as FormulaNodeData).inputTypes?.[index] : undefined;
      if (!type || !isInlineInputType(type)) continue;
      const indexes = result.get(edge.target) ?? [];
      indexes.push(index);
      result.set(edge.target, indexes);
    }
    return result;
  }, [edges, nodes]);
  const incomingCanvasNodes = useMemo(() => nodes.map((node) => toFlowNode(
    node,
    {
      ...callbacks,
      selectedSlot: selectedSlot?.nodeId === node.id ? selectedSlot.index : null,
      boundInlineInputs: boundInlineByNode.get(node.id) ?? [],
    },
    selectedNodeId,
    editable,
    selectedIdSet,
  )), [
    boundInlineByNode,
    callbacks,
    editable,
    nodes,
    selectedIdSet,
    selectedNodeId,
    selectedSlot,
  ]);
  const [canvasNodes, setCanvasNodes] = useState<FormulaFlowNode[]>(incomingCanvasNodes);
  const hasMountedCanvas = useRef(false);
  useEffect(() => {
    if (!hasMountedCanvas.current) {
      hasMountedCanvas.current = true;
      return;
    }
    setCanvasNodes((current) => reconcileCanvasNodes(current, incomingCanvasNodes));
  }, [incomingCanvasNodes]);
  useEffect(() => {
    canvasNodesRef.current = canvasNodes;
  }, [canvasNodes]);
  const flowEdges = useMemo(
    () => edges.map((edge) => toFlowEdge(edge, selectedEdgeIdSet)),
    [edges, selectedEdgeIdSet],
  );
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
      className={`formula-canvas formula-canvas--${props.mode}${props.compact ? " formula-canvas--compact" : ""}`}
      role="region"
      aria-label={props.ariaLabel ?? (editable ? "Formula editor canvas" : "Read-only formula diagram")}
      aria-readonly={!editable}
      aria-keyshortcuts="0 Escape"
      tabIndex={0}
      onKeyDown={(event) => {
        if (isTextEntryTarget(event.target)) return;
        if (event.key === "0") {
          event.preventDefault();
          void flowInstance.current?.fitView({ padding: 0.16 });
        } else if (event.key === "Escape") {
          event.preventDefault();
          onSelectNode?.(null);
        }
      }}
      onDragOver={editable ? (event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      } : undefined}
    >
      <ReactFlow
        nodes={canvasNodes}
        edges={flowEdges}
        nodeTypes={NODE_TYPES}
        fitView
        minZoom={0.25}
        maxZoom={1.8}
        zoomOnScroll
        zoomOnPinch
        panOnDrag
        nodesDraggable={editable}
        nodesConnectable={editable}
        deleteKeyCode={null}
        onInit={(instance) => {
          const typed = instance as unknown as ReactFlowInstance;
          flowInstance.current = typed;
          onInit?.(typed);
        }}
        isValidConnection={editable ? valid : undefined}
        onConnect={editable ? (connection) => {
          if (valid(connection)) onConnect?.(connection);
        } : undefined}
        onNodesChange={(changes: NodeChange[]) => {
          setCanvasNodes((current) => applyCanvasNodeChanges(
            changes as NodeChange<FormulaFlowNode>[],
            current,
          ));
        }}
        onEdgesChange={editable ? (changes: EdgeChange[]) => {
          const persistedChanges = changes.filter((change) => change.type !== "select");
          if (persistedChanges.length > 0) {
            onEdgesChange?.(applyEdgeChanges(persistedChanges, flowEdges).map(fromFlowEdge));
          }
        } : undefined}
        onNodeClick={(_event, node) => {
          onClearSlot?.();
          onSelectNode?.(node.id);
        }}
        onNodeDragStart={editable ? () => onNodeDragStart?.() : undefined}
        onNodeDragStop={editable ? (_event, stoppedNode, draggedNodes) => {
          const moved = new Map(draggedNodes.map((node) => [node.id, node]));
          moved.set(stoppedNode.id, stoppedNode);
          const finalCanvasNodes = canvasNodesRef.current.map((node) => {
            const finalNode = moved.get(node.id);
            return finalNode ? {
              ...node,
              position: finalNode.position,
              dragging: false,
            } : node;
          });
          setCanvasNodes(finalCanvasNodes);
          canvasNodesRef.current = finalCanvasNodes;
          onNodeDragStop?.(commitCanvasPositions(nodes, finalCanvasNodes));
        } : undefined}
        onSelectionChange={onSelectionChange ? handleSelectionChange : undefined}
        onPaneClick={() => {
          onClearSlot?.();
          onSelectNode?.(null);
        }}
        onDrop={editable ? (event: DragEvent) => {
          event.preventDefault();
          const primitive = event.dataTransfer.getData("application/x-alphalineage-primitive");
          if (!primitive) return;
          const point = flowInstance.current?.screenToFlowPosition({
            x: event.clientX,
            y: event.clientY,
          });
          if (point) onDropCanvas?.(primitive, point, { exact: true, anchor: "center" });
        } : undefined}
      >
        <Background gap={20} size={1} />
      </ReactFlow>
    </div>
  );
}
