import type { Connection, ReactFlowInstance } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addFormula,
  deleteFormula,
  getCategories,
  getFormula,
  getFormulaImpact,
  getPrimitives,
  listFormulaResults,
  listFormulas,
  putCategories,
  setPrimitiveCategory,
  updateFormula,
  validateFormula,
} from "../api/client";
import type {
  FactorNode,
  FormulaDetail,
  FormulaDraft,
  FormulaDraftEdge,
  FormulaDraftNode,
  FormulaImpact,
  FormulaInputSpec,
  FormulaResult,
  FormulaSpec,
  FormulaTestSource,
  PrimitiveInfo,
} from "../api/types";
import { FormulaBacktestDrawer } from "./FormulaBacktestDrawer";
import { FormulaCanvas } from "./FormulaCanvas";
import {
  autoLayoutGraph,
  blankFormulaGraph,
  bodyToFormulaGraph,
  deleteFormulaSelection,
  findAvailableNodePosition,
  findAvailableSubgraphPosition,
  formulaNodeHeight,
  graphToFormulaBody,
  isInlineInputType,
  nodeOutputType,
  OUTPUT_NODE_ID,
  primitiveNode,
  removeFormulaInput,
  repairFormulaGraph,
  targetInputType,
  typesCompatible,
  type FormulaGraph,
  type FormulaNodeData,
} from "./formulaGraph";
import { parseFormula, serializeFormula } from "./formulaText";

const TYPE_OPTIONS = ["series", "signal", "window", "scalar", "bool"];
const DEFAULT_INPUTS: FormulaInputSpec[] = [];

const DEMO_PRIMITIVES: PrimitiveInfo[] = [
  { name: "open", display_name: "Open", description: "Opening price for each symbol and date.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "high", display_name: "High", description: "Highest traded price for each symbol and date.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "low", display_name: "Low", description: "Lowest traded price for each symbol and date.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "close", display_name: "Close", description: "Closing price for each symbol and date.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "volume", display_name: "Volume", description: "Trading volume for each symbol and date.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "vwap", display_name: "HLC3", description: "Typical price derived from high, low and close.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "returns", display_name: "Returns", description: "Single-period return derived from closing prices.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "rank", display_name: "Cross-sectional rank", description: "Ranks symbols against one another on each date.", kind: "operator", arg_types: ["series"], inputs: [{ name: "series", type: "series", description: "Series to rank." }], out_type: "signal", user: false, origin: "builtin", category: "cross_sectional" },
  { name: "ts_mean", display_name: "Moving average", description: "Trailing arithmetic mean over a lookback window.", kind: "operator", arg_types: ["series", "window"], inputs: [{ name: "series", type: "series", description: "Series to average." }, { name: "lookback", type: "window", description: "Trailing period count." }], out_type: "series", user: false, origin: "builtin", category: "time_series" },
  { name: "sub", display_name: "Subtract", description: "Subtracts the second series from the first.", kind: "operator", arg_types: ["series", "series"], inputs: [{ name: "left", type: "series", description: "Series to subtract from." }, { name: "right", type: "series", description: "Series to subtract." }], out_type: "series", user: false, origin: "builtin", category: "arithmetic" },
  { name: "window", display_name: "Window", description: "A whole-number lookback.", kind: "ephemeral", arg_types: [], inputs: [], out_type: "window", user: false, origin: "value", category: "constant" },
  { name: "const", display_name: "Scalar", description: "A numeric constant.", kind: "ephemeral", arg_types: [], inputs: [], out_type: "scalar", user: false, origin: "value", category: "constant" },
];

interface PendingInsert {
  primitiveName: string;
  nodeId: string;
  inputIndex: number;
}

function humanCategory(value: string): string {
  return value.replace(/_/g, " ");
}

function draftGraph(draft: FormulaDraft | undefined, inputs: FormulaInputSpec[], outType: string): FormulaGraph {
  if (draft?.graphNodes?.length) {
    return { nodes: draft.graphNodes, edges: draft.graphEdges ?? [] };
  }
  return blankFormulaGraph(inputs, outType);
}

function resultOutputType(result: FormulaResult, primitives: PrimitiveInfo[]): string {
  if (result.out_type) return result.out_type;
  const root = result.expanded_tree ?? result.tree;
  return primitives.find((primitive) => primitive.name === root.name)?.out_type ?? "signal";
}

function isPrimitive(source: PrimitiveInfo | FormulaResult): source is PrimitiveInfo {
  return "arg_types" in source;
}

function formulaAsPrimitive(formula: FormulaSpec): PrimitiveInfo {
  return {
    name: formula.runtime_name ?? formula.name,
    logical_name: formula.name,
    runtime_name: formula.runtime_name ?? formula.name,
    display_name: formula.display_name || formula.name.replace(/_/g, " "),
    description: formula.description,
    kind: formula.arg_types.length ? "operator" : "operand",
    arg_types: formula.arg_types,
    inputs: formula.inputs,
    out_type: formula.out_type,
    user: true,
    origin: formula.origin ?? "user_formula",
    editable: formula.editable ?? true,
    category: formula.category ?? "custom",
    revision: formula.revision,
    family: formula.family,
    aliases: formula.aliases,
    catalog_revision: formula.catalog_revision,
  };
}

function matchesSearch(query: string, ...values: Array<string | undefined | null>): boolean {
  return !query || values.some((value) => value?.toLowerCase().includes(query));
}

function formatMetric(value: number | boolean | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

function sameIds(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((id, index) => id === right[index]);
}

function editableCopyName(value: string): string {
  let stem = value.trim().toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
  if (!stem || !/^[a-z]/.test(stem)) stem = `formula_${stem || "result"}`;
  stem = stem.slice(0, 59).replace(/_+$/g, "") || "formula_result";
  return `${stem}_copy`;
}

export function FormulaEditorPage({
  formulaDraft,
  onFormulaDraftChange,
  defaultUniverse,
  onDataSync,
  canSubmit = true,
}: {
  formulaDraft?: FormulaDraft;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
  defaultUniverse?: string;
  onDataSync?: () => void;
  canSubmit?: boolean;
}) {
  const initialInputs = formulaDraft?.inputs ?? DEFAULT_INPUTS;
  const initialOutType = formulaDraft?.out_type ?? "signal";
  const initialGraph = repairFormulaGraph(
    draftGraph(formulaDraft, initialInputs, initialOutType).nodes,
    draftGraph(formulaDraft, initialInputs, initialOutType).edges,
    initialInputs,
    initialOutType,
  );
  const defaultName = "my_formula";
  const defaultDisplayName = "My formula";
  const defaultCategory = "custom";

  const [primitives, setPrimitives] = useState<PrimitiveInfo[]>(canSubmit ? [] : DEMO_PRIMITIVES);
  const [formulas, setFormulas] = useState<FormulaSpec[]>([]);
  const [results, setResults] = useState<FormulaResult[]>([]);
  const [categories, setCategories] = useState<string[]>(["custom"]);
  const [name, setName] = useState(formulaDraft?.name ?? defaultName);
  const [displayName, setDisplayName] = useState(formulaDraft?.display_name ?? defaultDisplayName);
  const [description, setDescription] = useState(formulaDraft?.description ?? "");
  const [inputs, setInputs] = useState<FormulaInputSpec[]>(initialInputs);
  const [outType, setOutType] = useState(initialOutType);
  const [category, setCategory] = useState(formulaDraft?.category ?? defaultCategory);
  const [nodes, setNodes] = useState<FormulaDraftNode[]>(initialGraph.nodes);
  const [edges, setEdges] = useState<FormulaDraftEdge[]>(initialGraph.edges);
  const [expression, setExpression] = useState(formulaDraft?.expression ?? "");
  const [activeMode, setActiveMode] = useState<"visual" | "expression">(formulaDraft?.activeMode ?? "visual");
  const [mobilePane, setMobilePane] = useState<"library" | "canvas" | "inspector">("canvas");
  const [libraryCollapsed, setLibraryCollapsed] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [canvasRevision, setCanvasRevision] = useState(0);
  const [loadedName, setLoadedName] = useState<string | null>(formulaDraft?.loadedName ?? null);
  const [loadedRevision, setLoadedRevision] = useState<number | null>(formulaDraft?.loadedRevision ?? null);
  const [formulaDetail, setFormulaDetail] = useState<FormulaDetail | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(formulaDraft?.selectedNodeId ?? null);
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>(formulaDraft?.selectedNodeId ? [formulaDraft.selectedNodeId] : []);
  const [selectedEdgeIds, setSelectedEdgeIds] = useState<string[]>([]);
  const [selectedSlot, setSelectedSlot] = useState<{ nodeId: string; index: number } | null>(null);
  const [selectedSource, setSelectedSource] = useState<PrimitiveInfo | FormulaResult | null>(null);
  const [search, setSearch] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [pendingInsert, setPendingInsert] = useState<PendingInsert | null>(null);
  const [pendingSave, setPendingSave] = useState<{ spec: FormulaSpec; impact: FormulaImpact } | null>(null);
  const [past, setPast] = useState<FormulaGraph[]>([]);
  const [future, setFuture] = useState<FormulaGraph[]>([]);
  const [backtestOpen, setBacktestOpen] = useState(false);
  const [backtestSource, setBacktestSource] = useState<FormulaTestSource | null>(null);
  const flowRef = useRef<ReactFlowInstance | null>(null);
  const stageRef = useRef<HTMLElement | null>(null);
  const clipboard = useRef<FormulaGraph | null>(null);
  const dragStart = useRef<FormulaGraph | null>(null);
  const inlineEditStart = useRef<FormulaGraph | null>(null);
  const inlineCommitTimer = useRef<number | null>(null);
  const nodeCounter = useRef(nodes.length + 1);
  const shouldSeedGraph = useRef(!formulaDraft?.graphNodes?.length);
  const seedBody = useRef(formulaDraft?.body);

  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedData = selectedNode?.data as FormulaNodeData | undefined;
  const selectedFormulaUpdate = (selectedData?.origin === "user_formula" || selectedData?.origin === "catalog_formula") && selectedData.logicalName
    ? formulas.find((formula) => (
      formula.name === selectedData.logicalName && formula.registered !== false && !formula.error &&
      (formula.revision ?? 1) > (selectedData.revision ?? 1)
    )) ?? null
    : null;

  const handleSelectionChange = useCallback((nodeIds: string[], edgeIds: string[]) => {
    setSelectedNodeIds((current) => sameIds(current, nodeIds) ? current : nodeIds);
    setSelectedEdgeIds((current) => sameIds(current, edgeIds) ? current : edgeIds);
    setSelectedNodeId((current) => {
      const next = nodeIds[nodeIds.length - 1] ?? null;
      return current === next ? current : next;
    });
    if (nodeIds.length || edgeIds.length) setSelectedSource(null);
  }, []);

  function refresh() {
    if (!canSubmit) return;
    Promise.all([getPrimitives(), listFormulas(), listFormulaResults(), getCategories()])
      .then(([nextPrimitives, nextFormulas, nextResults, nextCategories]) => {
        setPrimitives(nextPrimitives);
        setFormulas(nextFormulas);
        setResults(nextResults);
        setCategories([...new Set([...nextCategories.order, "custom"])]);
      })
      .catch((reason) => setError(String(reason)));
  }

  useEffect(refresh, [canSubmit]);

  useEffect(() => {
    if (shouldSeedGraph.current && primitives.length && edges.length === 0) {
      shouldSeedGraph.current = false;
      if (seedBody.current) {
        const graph = bodyToFormulaGraph(seedBody.current, inputs, primitives, outType);
        setNodes(graph.nodes);
        setEdges(graph.edges);
        setExpression(serializeFormula(seedBody.current, inputs));
        setCanvasRevision((revision) => revision + 1);
        return;
      }
    }
  }, [primitives]);

  useEffect(() => {
    onFormulaDraftChange?.({
      name,
      display_name: displayName,
      description,
      arg_types: inputs.map((input) => input.type),
      inputs,
      out_type: outType,
      category,
      expression,
      activeMode,
      loadedName,
      loadedRevision,
      graphNodes: nodes,
      graphEdges: edges,
      selectedNodeId,
    });
  }, [activeMode, category, description, displayName, edges, expression, inputs, loadedName, loadedRevision, name, nodes, onFormulaDraftChange, outType, selectedNodeId]);

  useEffect(() => {
    if (!selectedSlot) return;
    const target = nodes.find((node) => node.id === selectedSlot.nodeId);
    if (!target || !targetInputType(target, `input-${selectedSlot.index}`)) setSelectedSlot(null);
  }, [nodes, selectedSlot]);

  useEffect(() => {
    if (activeMode !== "visual") return;
    const frame = requestAnimationFrame(() => {
      void flowRef.current?.fitView({ padding: 0.18, duration: 180 });
    });
    return () => cancelAnimationFrame(frame);
  }, [activeMode, canvasRevision]);

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  useEffect(() => () => {
    if (inlineCommitTimer.current != null) window.clearTimeout(inlineCommitTimer.current);
  }, []);

  function graphSnapshot(): FormulaGraph {
    return { nodes, edges };
  }

  function finishInlineEdit() {
    if (inlineCommitTimer.current != null) {
      window.clearTimeout(inlineCommitTimer.current);
      inlineCommitTimer.current = null;
    }
    const previous = inlineEditStart.current;
    inlineEditStart.current = null;
    if (!previous) return;
    setPast((history) => [...history.slice(-39), previous]);
    setFuture([]);
  }

  function commitGraph(next: FormulaGraph, contractInputs = inputs, contractOutType = outType) {
    finishInlineEdit();
    const repaired = repairFormulaGraph(next.nodes, next.edges, contractInputs, contractOutType);
    setPast((history) => [...history.slice(-39), graphSnapshot()]);
    setFuture([]);
    setNodes(repaired.nodes);
    setEdges(repaired.edges);
    setDirty(true);
    try {
      setExpression(serializeFormula(graphToFormulaBody(repaired.nodes, repaired.edges), contractInputs));
      setParseError(null);
    } catch {
      // Incomplete visual drafts are expected while blocks are being assembled.
    }
  }

  function finishDrag(nextNodes: FormulaDraftNode[]) {
    const previous = dragStart.current;
    dragStart.current = null;
    if (!previous) return;
    const repaired = repairFormulaGraph(nextNodes, edges, inputs, outType);
    const moved = repaired.nodes.some((node) => {
      const before = previous.nodes.find((candidate) => candidate.id === node.id);
      return before && (before.x !== node.x || before.y !== node.y);
    });
    if (!moved) return;
    setPast((history) => [...history.slice(-39), previous]);
    setFuture([]);
    setNodes(repaired.nodes);
    setEdges(repaired.edges);
    setDirty(true);
  }

  function changeInlineValue(nodeId: string, index: number, value: number | null) {
    if (!inlineEditStart.current) inlineEditStart.current = graphSnapshot();
    const nextNodes = nodes.map((node) => node.id === nodeId ? {
      ...node,
      data: {
        ...node.data,
        inlineValues: {
          ...((node.data as FormulaNodeData).inlineValues ?? {}),
          [String(index)]: value,
        },
      },
    } : node);
    setNodes(nextNodes);
    setDirty(true);
    try {
      setExpression(serializeFormula(graphToFormulaBody(nextNodes, edges), inputs));
      setParseError(null);
    } catch {
      // Blank numeric parameters are a valid editing state.
    }
    if (inlineCommitTimer.current != null) window.clearTimeout(inlineCommitTimer.current);
    inlineCommitTimer.current = window.setTimeout(finishInlineEdit, 450);
  }

  function undo() {
    const previous = past[past.length - 1];
    if (!previous) return;
    setFuture((items) => [graphSnapshot(), ...items]);
    setPast((items) => items.slice(0, -1));
    const repaired = repairFormulaGraph(previous.nodes, previous.edges, inputs, outType);
    setNodes(repaired.nodes);
    setEdges(repaired.edges);
    setSelectedSlot(null);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
    setDirty(true);
    try {
      setExpression(serializeFormula(graphToFormulaBody(repaired.nodes, repaired.edges), inputs));
      setParseError(null);
    } catch {
      // An incomplete historical draft stays editable in visual mode.
    }
  }

  function redo() {
    const next = future[0];
    if (!next) return;
    setPast((items) => [...items, graphSnapshot()]);
    setFuture((items) => items.slice(1));
    const repaired = repairFormulaGraph(next.nodes, next.edges, inputs, outType);
    setNodes(repaired.nodes);
    setEdges(repaired.edges);
    setSelectedSlot(null);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
    setDirty(true);
    try {
      setExpression(serializeFormula(graphToFormulaBody(repaired.nodes, repaired.edges), inputs));
      setParseError(null);
    } catch {
      // An incomplete historical draft stays editable in visual mode.
    }
  }

  function layoutAndFit() {
    commitGraph(autoLayoutGraph(nodes, edges));
    requestAnimationFrame(() => {
      void flowRef.current?.fitView({ padding: 0.18, duration: 180 });
    });
  }

  function updateOutputType(value: string) {
    const repaired = repairFormulaGraph(nodes, edges, inputs, value);
    setOutType(value);
    commitGraph(repaired, inputs, value);
  }

  function updateInput(index: number, patch: Partial<FormulaInputSpec>) {
    const nextInputs = inputs.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item);
    setInputs(nextInputs);
    commitGraph({ nodes, edges }, nextInputs);
  }

  function addInput() {
    const index = inputs.length;
    const input = { name: `input_${index + 1}`, type: "series", description: "Formula input." };
    const nextInputs = [...inputs, input];
    setInputs(nextInputs);
    commitGraph({ nodes, edges }, nextInputs);
  }

  function removeInput(index: number) {
    const inputNode = nodes.find((node) => (node.data as FormulaNodeData).kind === "input" && (node.data as FormulaNodeData).argIndex === index);
    if (inputNode && edges.some((edge) => edge.source === inputNode.id)) {
      setError("Disconnect this input before removing it.");
      return;
    }
    const nextInputs = inputs.filter((_, itemIndex) => itemIndex !== index);
    const graph = removeFormulaInput(nodes, edges, index, nextInputs, outType);
    setInputs(nextInputs);
    commitGraph(graph, nextInputs);
  }

  function nextNodeId(prefix = "formula-node"): string {
    const used = new Set(nodes.map((node) => node.id));
    let id = `${prefix}-${nodeCounter.current++}`;
    while (used.has(id)) id = `${prefix}-${nodeCounter.current++}`;
    return id;
  }

  function canvasCenter(): { x: number; y: number } {
    const bounds = stageRef.current?.getBoundingClientRect();
    if (bounds && flowRef.current) {
      return flowRef.current.screenToFlowPosition({
        x: bounds.left + bounds.width / 2,
        y: bounds.top + bounds.height / 2,
      });
    }
    return { x: 260, y: 180 };
  }

  function nextNodeFor(source: PrimitiveInfo | FormulaResult, point = { x: 260, y: 180 }, exact = false): FormulaDraftNode {
    const id = nextNodeId();
    const candidate: FormulaDraftNode = isPrimitive(source) ? primitiveNode(source, id, point.x, point.y) : {
      id,
      type: "formula",
      x: point.x,
      y: point.y,
      data: {
        kind: "factor",
        label: source.name,
        description: source.notes || "Saved alpha factor snapshot.",
        origin: "saved_factor",
        outType: resultOutputType(source, primitives),
        factorId: source.id,
        factorBody: source.expanded_tree ?? source.tree,
        locked: true,
      } satisfies FormulaNodeData,
    };
    if (exact) return candidate;
    const available = findAvailableNodePosition(nodes, point, candidate);
    return { ...candidate, x: available.x, y: available.y };
  }

  function findSource(key: string): PrimitiveInfo | FormulaResult | undefined {
    if (key.startsWith("result:")) return results.find((result) => result.id === key.slice(7));
    if (key.startsWith("formula:")) {
      const logicalName = key.slice("formula:".length);
      const latest = formulas.find((formula) => formula.name === logicalName && formula.registered !== false && !formula.error);
      return latest ? formulaAsPrimitive(latest) : undefined;
    }
    return primitives.find((primitive) => primitive.name === key || primitive.logical_name === key);
  }

  function insertOnCanvas(key: string, point?: { x: number; y: number }) {
    const source = findSource(key);
    if (!source) return;
    if (selectedSlot && !point) {
      insertIntoSlot(selectedSlot.nodeId, selectedSlot.index, key);
      return;
    }
    let node = nextNodeFor(source, point ?? canvasCenter(), Boolean(point));
    if (point) node = { ...node, x: point.x - 88, y: point.y - formulaNodeHeight(node) / 2 };
    commitGraph({ nodes: [...nodes, node], edges });
    setSelectedSlot(null);
    setSelectedNodeId(node.id);
    setSelectedNodeIds([node.id]);
    setSelectedEdgeIds([]);
  }

  function insertIntoSlot(nodeId: string, inputIndex: number, key: string) {
    const source = findSource(key);
    const target = nodes.find((node) => node.id === nodeId);
    if (!source || !target) return;
    const candidate = nextNodeFor(source, { x: target.x - 230, y: target.y + inputIndex * 40 });
    const expected = targetInputType(target, `input-${inputIndex}`);
    if (!expected || !typesCompatible(nodeOutputType(candidate), expected)) {
      setError(`${candidate.data.label} produces ${nodeOutputType(candidate)}, but this input expects ${expected}.`);
      return;
    }
    const occupied = edges.some((edge) => edge.target === nodeId && edge.targetHandle === `input-${inputIndex}`);
    if (occupied) {
      setPendingInsert({ primitiveName: key, nodeId, inputIndex });
      return;
    }
    commitGraph({
      nodes: [...nodes, candidate],
      edges: [...edges, {
        id: `${candidate.id}-${nodeId}-input-${inputIndex}`,
        source: candidate.id,
        target: nodeId,
        sourceHandle: "output",
        targetHandle: `input-${inputIndex}`,
      }],
    });
    setSelectedSlot(null);
    setSelectedNodeId(candidate.id);
    setSelectedNodeIds([candidate.id]);
    setSelectedEdgeIds([]);
  }

  function resolveOccupied(mode: "wrap" | "replace") {
    if (!pendingInsert) return;
    const { primitiveName, nodeId, inputIndex } = pendingInsert;
    const source = findSource(primitiveName);
    const target = nodes.find((node) => node.id === nodeId);
    const existing = edges.find((edge) => edge.target === nodeId && edge.targetHandle === `input-${inputIndex}`);
    if (!source || !target || !existing) return;
    const candidate = nextNodeFor(source, { x: target.x - 230, y: target.y + inputIndex * 50 });
    const remaining = edges.filter((edge) => edge.id !== existing.id);
    const nextEdges: FormulaDraftEdge[] = [...remaining, {
      id: `${candidate.id}-${nodeId}-input-${inputIndex}`,
      source: candidate.id,
      target: nodeId,
      sourceHandle: "output",
      targetHandle: `input-${inputIndex}`,
    }];
    if (mode === "wrap") {
      const oldSource = nodes.find((node) => node.id === existing.source);
      const data = candidate.data as FormulaNodeData;
      const compatibleIndex = oldSource
        ? (data.inputTypes ?? []).findIndex((type) => typesCompatible(nodeOutputType(oldSource), type))
        : -1;
      if (compatibleIndex < 0) {
        setError(`${candidate.data.label} cannot wrap the existing value because it has no compatible input.`);
        return;
      }
      nextEdges.push({
        id: `${existing.source}-${candidate.id}-input-${compatibleIndex}`,
        source: existing.source,
        target: candidate.id,
        sourceHandle: "output",
        targetHandle: `input-${compatibleIndex}`,
      });
    }
    commitGraph({ nodes: [...nodes, candidate], edges: nextEdges });
    setPendingInsert(null);
    setSelectedSlot(null);
  }

  function connect(connection: Connection) {
    if (!connection.source || !connection.target || !connection.targetHandle) return;
    commitGraph({
      nodes,
      edges: [...edges, {
        id: `${connection.source}-${connection.target}-${connection.targetHandle}`,
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle,
        targetHandle: connection.targetHandle,
      }],
    });
  }

  function bindInlineInput(nodeId: string, inputIndex: number, sourceId: string) {
    const target = nodes.find((node) => node.id === nodeId);
    const expected = target ? targetInputType(target, `input-${inputIndex}`) : null;
    if (!target || !expected || !isInlineInputType(expected)) return;
    const remaining = edges.filter((edge) => !(
      edge.target === nodeId && edge.targetHandle === `input-${inputIndex}`
    ));
    if (!sourceId) {
      const data = target.data as FormulaNodeData;
      const current = data.inlineValues?.[String(inputIndex)];
      const nextNodes = current == null ? nodes.map((node) => node.id === nodeId ? {
        ...node,
        data: {
          ...node.data,
          inlineValues: {
            ...(data.inlineValues ?? {}),
            [String(inputIndex)]: expected === "window" ? 20 : 1,
          },
        },
      } : node) : nodes;
      commitGraph({ nodes: nextNodes, edges: remaining });
      return;
    }
    const source = nodes.find((node) => node.id === sourceId);
    if (!source || (source.data as FormulaNodeData).kind !== "input" ||
        !typesCompatible(nodeOutputType(source), expected)) return;
    commitGraph({
      nodes,
      edges: [...remaining, {
        id: `${sourceId}-${nodeId}-input-${inputIndex}`,
        source: sourceId,
        target: nodeId,
        sourceHandle: "output",
        targetHandle: `input-${inputIndex}`,
      }],
    });
  }

  function updateSelectedFormulaReference(latest: FormulaSpec) {
    if (!selectedNode || (selectedData?.origin !== "user_formula" && selectedData?.origin !== "catalog_formula")) return;
    const oldData = selectedNode.data as FormulaNodeData;
    const replacement = primitiveNode(formulaAsPrimitive(latest), selectedNode.id, selectedNode.x, selectedNode.y);
    const nextData = replacement.data as FormulaNodeData;
    const oldTypes = oldData.inputTypes ?? [];
    const oldNames = oldData.inputNames ?? [];
    const nextTypes = nextData.inputTypes ?? [];
    const nextNames = nextData.inputNames ?? [];
    const usedNextIndexes = new Set<number>();
    const portMap = new Map<number, number>();

    oldTypes.forEach((oldType, oldIndex) => {
      const namedIndex = nextNames.findIndex((name, index) => (
        !usedNextIndexes.has(index) && name === oldNames[oldIndex] && typesCompatible(oldType, nextTypes[index])
      ));
      const sameIndex = oldIndex < nextTypes.length && !usedNextIndexes.has(oldIndex) &&
        typesCompatible(oldType, nextTypes[oldIndex]) ? oldIndex : -1;
      const nextIndex = namedIndex >= 0 ? namedIndex : sameIndex;
      if (nextIndex >= 0) {
        portMap.set(oldIndex, nextIndex);
        usedNextIndexes.add(nextIndex);
      }
    });

    const inlineValues = { ...(nextData.inlineValues ?? {}) };
    for (const [oldIndex, nextIndex] of portMap) {
      const value = oldData.inlineValues?.[String(oldIndex)];
      if (isInlineInputType(nextTypes[nextIndex]) && value !== undefined) {
        inlineValues[String(nextIndex)] = value;
      }
    }
    replacement.data = { ...replacement.data, inlineValues };

    const occupiedTargets = new Set<string>();
    const nextEdges: FormulaDraftEdge[] = [];
    for (const edge of edges) {
      if (edge.target === selectedNode.id) {
        const match = edge.targetHandle?.match(/^input-(\d+)$/);
        const oldIndex = match ? Number(match[1]) : -1;
        const nextIndex = portMap.get(oldIndex);
        const source = nodes.find((node) => node.id === edge.source);
        const expected = nextIndex == null ? null : nextTypes[nextIndex];
        const targetKey = `${selectedNode.id}:input-${nextIndex}`;
        if (nextIndex == null || !source || !expected || occupiedTargets.has(targetKey) ||
            !typesCompatible(nodeOutputType(source), expected)) continue;
        occupiedTargets.add(targetKey);
        nextEdges.push({
          ...edge,
          id: `${edge.source}-${selectedNode.id}-input-${nextIndex}`,
          targetHandle: `input-${nextIndex}`,
        });
        continue;
      }
      if (edge.source === selectedNode.id) {
        const target = nodes.find((node) => node.id === edge.target);
        const expected = target ? targetInputType(target, edge.targetHandle) : null;
        if (!expected || !typesCompatible(nextData.outType, expected)) continue;
      }
      nextEdges.push(edge);
    }

    commitGraph({
      nodes: nodes.map((node) => node.id === selectedNode.id ? replacement : node),
      edges: nextEdges,
    });
    setMessage(`Updated ${latest.display_name || latest.name} to v${latest.revision ?? 1}.`);
  }

  function copySelected() {
    const ids = new Set(selectedNodeIds.length ? selectedNodeIds : selectedNode ? [selectedNode.id] : []);
    const copiedNodes = nodes.filter((node) => ids.has(node.id) && node.id !== OUTPUT_NODE_ID && (node.data as FormulaNodeData).kind !== "input")
      .map((node) => ({ ...node, data: { ...node.data } }));
    if (!copiedNodes.length) return;
    const copiedIds = new Set(copiedNodes.map((node) => node.id));
    clipboard.current = {
      nodes: copiedNodes,
      edges: edges.filter((edge) => copiedIds.has(edge.source) && copiedIds.has(edge.target)).map((edge) => ({ ...edge })),
    };
  }

  function pasteCopied() {
    const copied = clipboard.current;
    if (!copied?.nodes.length) return;
    const idMap = new Map<string, string>();
    const rawCopies = copied.nodes.map((source) => {
      const node = { ...source, id: nextNodeId(), x: source.x + 35, y: source.y + 35, data: { ...source.data } };
      idMap.set(source.id, node.id);
      return node;
    });
    const pasted = findAvailableSubgraphPosition(nodes, rawCopies);
    const pastedEdges = copied.edges.map((edge, index) => ({
      ...edge,
      id: `pasted-${nodeCounter.current}-${index}`,
      source: idMap.get(edge.source)!,
      target: idMap.get(edge.target)!,
    }));
    commitGraph({ nodes: [...nodes, ...pasted], edges: [...edges, ...pastedEdges] });
    const ids = pasted.map((node) => node.id);
    setSelectedNodeId(ids[ids.length - 1] ?? null);
    setSelectedNodeIds(ids);
    setSelectedEdgeIds([]);
    clipboard.current = { nodes: pasted, edges: pastedEdges };
  }

  function deleteSelected() {
    const requested = new Set(selectedNodeIds.length ? selectedNodeIds : selectedNode ? [selectedNode.id] : []);
    const next = deleteFormulaSelection(nodes, edges, requested, selectedEdgeIds);
    if (next.nodes.length === nodes.length && next.edges.length === edges.length) return;
    commitGraph(next);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
  }

  function shortcutTarget(target: EventTarget | null): boolean {
    if (document.querySelector("[role='dialog']")) return true;
    const element = target instanceof HTMLElement ? target : null;
    return Boolean(element?.closest("input, textarea, select, [contenteditable='true'], [role='dialog']"));
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (shortcutTarget(event.target)) return;
      const command = event.ctrlKey || event.metaKey;
      const key = event.key.toLowerCase();
      if (command && key === "s") {
        event.preventDefault();
        void save();
      } else if (command && key === "z" && event.shiftKey) {
        event.preventDefault();
        redo();
      } else if ((command && key === "y") || (command && key === "z" && !event.shiftKey)) {
        event.preventDefault();
        if (key === "y") redo(); else undo();
      } else if (command && key === "c") {
        event.preventDefault();
        copySelected();
      } else if (command && key === "v") {
        event.preventDefault();
        pasteCopied();
      } else if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        deleteSelected();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  function expandSelectedFactor() {
    if (!selectedNode || selectedData?.kind !== "factor" || !selectedData.factorBody) return;
    const subgraph = bodyToFormulaGraph(selectedData.factorBody, [], primitives, selectedData.outType);
    const outputEdge = subgraph.edges.find((edge) => edge.target === OUTPUT_NODE_ID);
    if (!outputEdge) return;
    const prefix = `${selectedNode.id}-expanded`;
    const idMap = new Map<string, string>();
    for (const node of subgraph.nodes) {
      if (node.id !== OUTPUT_NODE_ID) idMap.set(node.id, nextNodeId(prefix));
    }
    const subNodes = subgraph.nodes
      .filter((node) => node.id !== OUTPUT_NODE_ID)
      .map((node) => ({ ...node, id: idMap.get(node.id)!, x: node.x + selectedNode.x - 160, y: node.y + selectedNode.y - 80 }));
    const subEdges = subgraph.edges
      .filter((edge) => edge.target !== OUTPUT_NODE_ID)
      .map((edge) => ({ ...edge, id: `${prefix}${edge.id}`, source: idMap.get(edge.source)!, target: idMap.get(edge.target)! }));
    const rootId = idMap.get(outputEdge.source);
    if (!rootId) return;
    const outgoing = edges.filter((edge) => edge.source === selectedNode.id).map((edge) => ({ ...edge, id: `${rootId}-${edge.target}-${edge.targetHandle}`, source: rootId }));
    commitGraph({
      nodes: [...nodes.filter((node) => node.id !== selectedNode.id), ...subNodes],
      edges: [...edges.filter((edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id), ...subEdges, ...outgoing],
    });
    setSelectedNodeId(rootId);
    setSelectedNodeIds([rootId]);
    setSelectedEdgeIds([]);
  }

  function buildSpec(): FormulaSpec {
    const body = graphToFormulaBody(nodes, edges);
    return {
      name,
      display_name: displayName,
      description,
      arg_types: inputs.map((input) => input.type),
      inputs,
      out_type: outType,
      body,
      category,
      revision: loadedRevision ?? 1,
    };
  }

  async function persist(spec: FormulaSpec, strategy: "update" | "upgrade_references" = "update") {
    if (!categories.includes(category)) {
      await putCategories({ order: [...categories, category] });
      setCategories((items) => [...new Set([...items, category])]);
    }
    const saved = loadedName
      ? await updateFormula(loadedName, spec, strategy)
      : await addFormula(spec);
    setLoadedName(saved.name);
    setLoadedRevision(saved.revision ?? 1);
    setDirty(false);
    setMessage(`Saved ${saved.display_name || saved.name}${saved.revision && saved.revision > 1 ? ` v${saved.revision}` : ""}.`);
    setPendingSave(null);
    refresh();
  }

  async function save() {
    if (!canSubmit) return;
    setError(null);
    setMessage(null);
    try {
      const spec = buildSpec();
      const validation = await validateFormula(spec);
      if (!validation.ok) throw new Error(validation.error ?? "Formula is invalid.");
      if (!loadedName) {
        await persist(spec);
        return;
      }
      const impact = await getFormulaImpact(loadedName, spec);
      if (impact.change === "calculation" && impact.has_references) {
        setPendingSave({ spec, impact });
        return;
      }
      await persist(spec);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function openBacktest() {
    setError(null);
    try {
      const spec = buildSpec();
      if (spec.out_type !== "series" && spec.out_type !== "signal") {
        throw new Error("Backtests require a series or signal output.");
      }
      if (canSubmit) {
        const validation = await validateFormula(spec);
        if (!validation.ok) throw new Error(validation.error ?? "Formula is invalid.");
      }
      setBacktestSource({
        kind: "draft",
        body: spec.body,
        inputs: spec.inputs ?? [],
        out_type: spec.out_type,
      });
      setBacktestOpen(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function branchPending() {
    if (!pendingSave) return;
    const branchName = `${pendingSave.spec.name}_branch`;
    setName(branchName);
    setDisplayName(`${pendingSave.spec.display_name} branch`);
    setLoadedName(null);
    setLoadedRevision(null);
    setPendingSave(null);
    setDirty(true);
    setMessage(`Editing a new branch named ${branchName}. Save when ready.`);
  }

  function loadFormula(spec: FormulaSpec) {
    const nextInputs = spec.inputs?.length
      ? spec.inputs
      : spec.arg_types.map((type, index) => ({ name: `input_${index + 1}`, type, description: "Formula input." }));
    const graph = bodyToFormulaGraph(spec.body, nextInputs, primitives, spec.out_type);
    setName(spec.name);
    setDisplayName(spec.display_name || spec.name.replace(/_/g, " "));
    setDescription(spec.description ?? "");
    setInputs(nextInputs);
    setOutType(spec.out_type);
    setCategory(spec.category || "custom");
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setExpression(serializeFormula(spec.body, nextInputs));
    setLoadedName(spec.name);
    setLoadedRevision(spec.revision ?? 1);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
    setSelectedSlot(null);
    setSelectedSource(null);
    setPast([]);
    setFuture([]);
    setCanvasRevision((revision) => revision + 1);
    setDirty(false);
    setMessage(null);
    setError(null);
    if (canSubmit) getFormula(spec.name).then(setFormulaDetail).catch(() => setFormulaDetail(null));
  }

  function newFormula() {
    if (dirty && !window.confirm("Discard the current unsaved draft?")) return;
    const graph = blankFormulaGraph(DEFAULT_INPUTS, "signal");
    setName(defaultName);
    setDisplayName(defaultDisplayName);
    setDescription("");
    setInputs(DEFAULT_INPUTS);
    setOutType("signal");
    setCategory(defaultCategory);
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setExpression("");
    setLoadedName(null);
    setLoadedRevision(null);
    setFormulaDetail(null);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
    setSelectedSlot(null);
    setSelectedSource(null);
    setPast([]);
    setFuture([]);
    setCanvasRevision((revision) => revision + 1);
    setDirty(false);
  }

  function openResultAsEditableCopy(result: FormulaResult) {
    if (dirty && !window.confirm("Discard the current unsaved draft and open this result as a copy?")) return;
    const body = result.expanded_tree ?? result.tree;
    const nextOutType = resultOutputType(result, primitives);
    const graph = bodyToFormulaGraph(body, [], primitives, nextOutType);
    const nextName = editableCopyName(result.name);
    setName(nextName);
    setDisplayName(`${result.name} copy`);
    setDescription(result.notes || `Editable copy of formula result ${result.name}.`);
    setInputs([]);
    setOutType(nextOutType);
    setCategory("custom");
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setExpression(serializeFormula(body, []));
    setActiveMode("visual");
    setLoadedName(null);
    setLoadedRevision(null);
    setFormulaDetail(null);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
    setSelectedSlot(null);
    setSelectedSource(null);
    setPast([]);
    setFuture([]);
    setCanvasRevision((revision) => revision + 1);
    setDirty(true);
    setMessage(`Opened ${result.name} as editable formula ${nextName}.`);
    setError(null);
  }

  function openFormulaAsEditableCopy(spec: FormulaSpec) {
    if (dirty && !window.confirm("Discard the current unsaved draft and open this starter as a copy?")) return;
    const nextInputs = spec.inputs?.length
      ? spec.inputs.map((input) => ({ ...input }))
      : spec.arg_types.map((type, index) => ({ name: `input_${index + 1}`, type, description: "Formula input." }));
    const graph = bodyToFormulaGraph(spec.body, nextInputs, primitives, spec.out_type);
    const baseName = spec.name.replace(/^ta_/, "");
    const nextName = editableCopyName(baseName);
    setName(nextName);
    setDisplayName(`${spec.display_name || baseName.replace(/_/g, " ")} copy`);
    setDescription(`Editable copy of the managed starter ${spec.display_name || spec.name}.`);
    setInputs(nextInputs);
    setOutType(spec.out_type);
    setCategory(spec.category || "technical_indicators");
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setExpression(serializeFormula(spec.body, nextInputs));
    setActiveMode("visual");
    setLoadedName(null);
    setLoadedRevision(null);
    setFormulaDetail(null);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
    setSelectedSlot(null);
    setSelectedSource(null);
    setPast([]);
    setFuture([]);
    setCanvasRevision((revision) => revision + 1);
    setDirty(true);
    setMessage(`Opened ${spec.display_name || spec.name} as editable formula ${nextName}.`);
    setError(null);
  }

  function useBuiltin(source: PrimitiveInfo) {
    if (!window.confirm(`Create a user-defined formula that starts from ${source.display_name ?? source.name}?`)) return;
    const nextInputs = (source.inputs ?? source.arg_types.map((type, index) => ({ name: `input_${index + 1}`, type, description: "Function input." }))).map((input) => ({ ...input }));
    const body: FactorNode = {
      name: source.runtime_name ?? source.name,
      children: nextInputs.map((_, index) => ({ name: "$arg", value: index })),
    };
    const graph = bodyToFormulaGraph(body, nextInputs, primitives, source.out_type);
    setName(`${source.logical_name ?? source.name}_custom`);
    setDisplayName(`${source.display_name ?? source.name} custom`);
    setDescription(`User-defined formula based on ${source.display_name ?? source.name}.`);
    setInputs(nextInputs);
    setOutType(source.out_type);
    setCategory("custom");
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setExpression(serializeFormula(body, nextInputs));
    setLoadedName(null);
    setLoadedRevision(null);
    setSelectedNodeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeIds([]);
    setSelectedSource(null);
    setSelectedSlot(null);
    setCanvasRevision((revision) => revision + 1);
    setDirty(true);
  }

  function editExpression(value: string) {
    setExpression(value);
    setDirty(true);
    const parsed = parseFormula(value, inputs, primitives);
    if (!parsed.tree) {
      setParseError(parsed.errors[0] ? `${parsed.errors[0].msg} at character ${parsed.errors[0].pos}` : "Invalid expression.");
      return;
    }
    setParseError(null);
    const graph = bodyToFormulaGraph(parsed.tree, inputs, primitives, outType);
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setSelectedSlot(null);
  }

  const query = search.trim().toLowerCase();
  const filteredPrimitives = useMemo(() => {
    return primitives.filter((primitive) => matchesSearch(query, primitive.name, primitive.display_name, primitive.description, primitive.category));
  }, [primitives, search]);

  const dataPrimitives = useMemo(() => filteredPrimitives.filter((primitive) => (
    primitive.origin === "data" || (primitive.kind === "operand" && primitive.origin !== "user_formula" && primitive.origin !== "catalog_formula")
  )), [filteredPrimitives]);

  const grouped = useMemo(() => {
    const groups = new Map<string, PrimitiveInfo[]>();
    for (const primitive of filteredPrimitives) {
      if (primitive.origin === "data" || primitive.origin === "value" || primitive.origin === "user_formula" || primitive.origin === "catalog_formula") continue;
      if (primitive.kind === "operand" || primitive.kind === "ephemeral") continue;
      const key = primitive.category ?? "uncategorized";
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(primitive);
    }
    const order = [...categories, ...[...groups.keys()].filter((key) => !categories.includes(key)).sort()];
    return order.flatMap((key) => groups.has(key) ? [[key, groups.get(key)!] as const] : []);
  }, [categories, filteredPrimitives]);

  const visibleFormulas = useMemo(() => formulas.filter((formula) => matchesSearch(
    query,
    formula.name,
    formula.display_name,
    formula.description,
    formula.category,
    formula.family,
    ...(formula.aliases ?? []),
    formula.error,
  )), [formulas, query]);

  const starterFormulas = useMemo(() => visibleFormulas.filter((formula) => (
    formula.origin === "catalog_formula"
  )), [visibleFormulas]);

  const userFormulas = useMemo(() => visibleFormulas.filter((formula) => (
    formula.origin !== "catalog_formula"
  )), [visibleFormulas]);

  const visibleResults = useMemo(() => results.filter((result) => matchesSearch(
    query,
    result.name,
    result.notes,
    result.provenance?.universe,
  )), [query, results]);

  return (
    <div className="formula-workspace" data-testid="formula-editor-page">
      <header className="formula-workspace__toolbar">
        <div>
          <h3>Formula Builder</h3>
          <p className="panel-note">Build from market data or other formulas, then save, reuse, and backtest the result.</p>
        </div>
        <div className="formula-toolbar__actions">
          <button type="button" onClick={newFormula}>New</button>
          <button type="button" onClick={save} className="primary-action" disabled={!canSubmit}>Save</button>
          <button type="button" onClick={openBacktest} disabled={!canSubmit}>Backtest</button>
          <details className="formula-toolbar__overflow">
            <summary aria-label="More formula actions">More</summary>
            <div>
              <button type="button" onClick={layoutAndFit}>Tidy layout</button>
            </div>
          </details>
        </div>
        <div className="formula-mode" role="tablist" aria-label="Formula editor mode">
          <button type="button" role="tab" aria-selected={activeMode === "visual"} onClick={() => setActiveMode("visual")}>Visual</button>
          <button type="button" role="tab" aria-selected={activeMode === "expression"} onClick={() => setActiveMode("expression")}>Expression</button>
        </div>
      </header>

      {!canSubmit && <p className="surface-message">Static demo mode keeps this workspace as a local draft. Connect the backend to validate and save formulas.</p>}
      {message && <p className="ok">{message}</p>}
      {error && <p className="error">{error}</p>}

      <ol className="formula-workspace__guide" aria-label="Formula builder steps">
        <li><strong>1.</strong> Start with Data, a function, or a saved formula.</li>
        <li><strong>2.</strong> Connect the calculation and edit numeric parameters inline.</li>
        <li><strong>3.</strong> Connect Formula output, then save or backtest.</li>
      </ol>
      <p className="formula-shortcuts hint">Shortcuts: Ctrl/Cmd+Z undo · Ctrl/Cmd+Shift+Z redo · Ctrl/Cmd+C/V copy/paste · Delete remove · Ctrl/Cmd+S save</p>

      <nav className="formula-mobile-nav" aria-label="Formula workspace panels">
        {(["library", "canvas", "inspector"] as const).map((pane) => (
          <button key={pane} type="button" aria-pressed={mobilePane === pane} onClick={() => setMobilePane(pane)}>{pane}</button>
        ))}
      </nav>

      <div className={`formula-workspace__grid formula-pane--${mobilePane}${libraryCollapsed ? " is-library-collapsed" : ""}${inspectorCollapsed ? " is-inspector-collapsed" : ""}`}>
        <aside className="formula-library" data-testid="formula-library">
          <header className="formula-panel-header">
            <h4>Building blocks</h4>
            <button type="button" aria-label={libraryCollapsed ? "Expand building blocks" : "Collapse building blocks"} onClick={() => setLibraryCollapsed((value) => !value)}>{libraryCollapsed ? "›" : "‹"}</button>
          </header>
          <label className="field">
            <span className="field-label">Find a building block</span>
            <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search functions and fields" />
          </label>
          <details open>
            <summary>Formula contract <span>required</span></summary>
            <div className="formula-library__items">
              <article className="formula-library__item formula-library__item--contract">
                <button type="button" className="formula-library__insert" onClick={() => {
                  setSelectedSource(null);
                  setSelectedSlot(null);
                  setSelectedNodeId(OUTPUT_NODE_ID);
                  setSelectedNodeIds([OUTPUT_NODE_ID]);
                  setSelectedEdgeIds([]);
                  setInspectorCollapsed(false);
                }}>
                  <strong>Formula output</strong>
                  <span>Every formula returns the block connected here.</span>
                  <code>{inputs.length ? `${inputs.length} exposed input(s)` : "market data / saved formulas"} {"->"} {outType}</code>
                </button>
              </article>
            </div>
          </details>

          {dataPrimitives.length > 0 && (
            <details open>
              <summary>Data <span>{dataPrimitives.length}</span></summary>
              <div className="formula-library__items">
                {dataPrimitives.map((item) => (
                  <article key={item.name} className="formula-library__item formula-library__item--data" draggable onDragStart={(event) => event.dataTransfer.setData("application/x-alphalineage-primitive", item.name)}>
                    <button type="button" className="formula-library__insert" onClick={() => insertOnCanvas(item.name)}><strong>{item.display_name ?? item.name}</strong><span>{item.description ?? "Market data field."}</span><code>Data {"->"} {item.out_type}</code></button>
                    <button type="button" className="ghost" onClick={() => { setSelectedSource(item); setSelectedSlot(null); }}>Inspect</button>
                  </article>
                ))}
              </div>
            </details>
          )}

          {starterFormulas.length > 0 && (
            <details open>
              <summary>Starter formulas <span>{starterFormulas.length}</span></summary>
              <div className="formula-library__items">
                {starterFormulas.map((formula) => {
                  const usable = formula.registered !== false && !formula.error;
                  return (
                    <article key={formula.name} className={`formula-library__item formula-library__item--catalog_formula${usable ? "" : " is-broken"}`} draggable={usable} onDragStart={(event) => { if (usable) event.dataTransfer.setData("application/x-alphalineage-primitive", `formula:${formula.name}`); }}>
                      <button type="button" className="formula-library__insert" disabled={!usable} onClick={() => insertOnCanvas(`formula:${formula.name}`)}>
                        <strong>{formula.display_name || formula.name}</strong>
                        <span>{formula.error ? `Unavailable: ${formula.error}` : formula.description || "Managed, reusable starter formula."}</span>
                        <code>{formula.inputs?.map((input) => input.name).join(", ") || "No inputs"} {"->"} {formula.out_type} · v{formula.revision ?? 1}</code>
                      </button>
                      <button type="button" className="ghost" onClick={() => openFormulaAsEditableCopy(formula)}>Open as editable copy</button>
                    </article>
                  );
                })}
              </div>
            </details>
          )}

          {userFormulas.length > 0 && (
            <details open>
              <summary>My formulas <span>{userFormulas.length}</span></summary>
              <div className="formula-library__items">
                {userFormulas.map((formula) => {
                  const usable = formula.registered !== false && !formula.error && formula.name !== loadedName;
                  return (
                    <article key={formula.name} className={`formula-library__item formula-library__item--user_formula${usable ? "" : " is-broken"}`} draggable={usable} onDragStart={(event) => { if (usable) event.dataTransfer.setData("application/x-alphalineage-primitive", `formula:${formula.name}`); }}>
                      <button type="button" className="formula-library__insert" disabled={!usable} onClick={() => insertOnCanvas(`formula:${formula.name}`)}>
                        <strong>{formula.display_name || formula.name}</strong>
                        <span>{formula.name === loadedName ? "Currently open; a formula cannot contain itself." : formula.error ? `Needs repair: ${formula.error}` : formula.description || "Reusable saved formula."}</span>
                        <code>{formula.inputs?.map((input) => input.name).join(", ") || "No inputs"} {"->"} {formula.out_type} · v{formula.revision ?? 1}</code>
                      </button>
                      <button type="button" className="ghost" onClick={() => {
                        loadFormula(formula);
                      }}>{formula.error ? "Repair" : "Edit"}</button>
                    </article>
                  );
                })}
              </div>
            </details>
          )}

          {grouped.map(([group, items]) => (
            <details key={group} open>
              <summary>{humanCategory(group)} <span>{items.length}</span></summary>
              <div className="formula-library__items">
                {items.map((item) => (
                  <article
                    key={item.name}
                    className={`formula-library__item formula-library__item--${item.origin ?? "builtin"}`}
                    draggable
                    onDragStart={(event) => event.dataTransfer.setData("application/x-alphalineage-primitive", item.name)}
                  >
                    <button type="button" className="formula-library__insert" onClick={() => insertOnCanvas(item.name)}>
                      <strong>{item.display_name ?? item.logical_name ?? item.name}</strong>
                      <span>{item.description ?? "Typed calculation block."}</span>
                      <code>{(item.inputs ?? []).map((input) => input.name).join(", ")} {"->"} {item.out_type}</code>
                    </button>
                    <button type="button" className="ghost" onClick={() => { setSelectedSource(item); setSelectedSlot(null); }}>Inspect</button>
                  </article>
                ))}
              </div>
            </details>
          ))}

          {visibleResults.length > 0 && (
            <details open>
              <summary>Formula Results <span>{visibleResults.length}</span></summary>
              <div className="formula-library__items">
                {visibleResults.map((result) => (
                  <article
                    key={result.id}
                    className="formula-library__item formula-library__item--factor"
                    draggable
                    onDragStart={(event) => event.dataTransfer.setData("application/x-alphalineage-primitive", `result:${result.id}`)}
                  >
                    <button type="button" className="formula-library__insert" onClick={() => insertOnCanvas(`result:${result.id}`)}>
                      <strong>{result.name}</strong>
                      <span>{result.notes || `Kept from ${result.provenance?.universe ?? result.universe ?? "a research run"}.`}</span>
                      <code>{result.kind === "backtest" ? "Backtest" : "Training"} snapshot {"->"} {resultOutputType(result, primitives)}</code>
                    </button>
                    <button type="button" className="ghost" onClick={() => { setSelectedSource(result); setSelectedSlot(null); }}>Inspect</button>
                  </article>
                ))}
              </div>
            </details>
          )}
        </aside>

        <main className="formula-stage" ref={stageRef}>
          {activeMode === "visual" ? (
            <FormulaCanvas
              key={canvasRevision}
              nodes={nodes}
              edges={edges}
              selectedNodeId={selectedNodeId}
              selectedNodeIds={selectedNodeIds}
              selectedEdgeIds={selectedEdgeIds}
              selectedSlot={selectedSlot}
              onNodesChange={setNodes}
              onNodeDragStart={() => {
                finishInlineEdit();
                dragStart.current = graphSnapshot();
              }}
              onNodeDragStop={finishDrag}
              onEdgesChange={(nextEdges) => commitGraph({ nodes, edges: nextEdges })}
              onConnect={connect}
              onSelectNode={(id) => {
                setSelectedNodeId(id);
                setSelectedNodeIds(id ? [id] : []);
                setSelectedEdgeIds([]);
                setSelectedSource(null);
              }}
              onSelectionChange={handleSelectionChange}
              onSelectSlot={(nodeId, index) => setSelectedSlot({ nodeId, index })}
              onClearSlot={() => setSelectedSlot(null)}
              onDropPrimitive={insertIntoSlot}
              onDropCanvas={insertOnCanvas}
              onValueChange={(nodeId, value) => commitGraph({ nodes: nodes.map((node) => node.id === nodeId ? { ...node, data: { ...node.data, value } } : node), edges })}
              onInlineValueChange={changeInlineValue}
              onInlineValueCommit={finishInlineEdit}
              onInit={(instance) => { flowRef.current = instance; }}
            />
          ) : (
            <section className="formula-expression-panel">
              <label className="field">
                <span className="field-label">Expression</span>
                <textarea rows={14} value={expression} onChange={(event) => editExpression(event.target.value)} aria-label="Formula expression" spellCheck={false} />
              </label>
              <p className="hint">Market fields are referenced directly, for example <code>rank(ts_mean(close, 20))</code>. Optional exposed inputs use a dollar sign.</p>
              {parseError && <p className="error">{parseError}</p>}
            </section>
          )}
        </main>

        <aside className="formula-inspector" data-testid="formula-inspector">
          <header className="formula-panel-header">
            <h4>Inspector</h4>
            <button type="button" aria-label={inspectorCollapsed ? "Expand inspector" : "Collapse inspector"} onClick={() => setInspectorCollapsed((value) => !value)}>{inspectorCollapsed ? "‹" : "›"}</button>
          </header>
          {selectedSource && isPrimitive(selectedSource) ? (
            <section>
              <span className="mode-chip">{selectedSource.origin?.replace("_", " ") ?? "built in"}</span>
              <h3>{selectedSource.display_name ?? selectedSource.name}</h3>
              <p>{selectedSource.description}</p>
              <dl className="formula-inspector__signature">
                {(selectedSource.inputs ?? []).map((input) => <div key={input.name}><dt>{input.name}: {input.type}</dt><dd>{input.description}</dd></div>)}
                <div><dt>Output</dt><dd>{selectedSource.out_type}</dd></div>
              </dl>
              {selectedSource.user ? (
                <button type="button" className="primary-action" onClick={() => {
                  const spec = formulas.find((formula) => formula.name === selectedSource.logical_name);
                  if (spec) loadFormula(spec);
                }}>Open formula</button>
              ) : (
                <>
                  <label className="field"><span className="field-label">Category</span><select value={selectedSource.category ?? "uncategorized"} onChange={async (event) => {
                    if (!canSubmit) return;
                    await setPrimitiveCategory(selectedSource.name, event.target.value);
                    refresh();
                  }}>{[...new Set([selectedSource.category ?? "uncategorized", ...categories])].map((item) => <option key={item} value={item}>{humanCategory(item)}</option>)}</select></label>
                  {selectedSource.kind === "operator" && <button type="button" className="primary-action" onClick={() => useBuiltin(selectedSource)}>Use as starting point</button>}
                </>
              )}
            </section>
          ) : selectedSource ? (
            <section>
              <span className="mode-chip">Formula result · {selectedSource.kind}</span>
              <h3>{selectedSource.name}</h3>
              <p>{selectedSource.notes || "Immutable calculation and research evidence."}</p>
              <dl className="formula-inspector__signature"><div><dt>Saved</dt><dd>{selectedSource.saved_at}</dd></div><div><dt>Universe</dt><dd>{selectedSource.provenance?.universe ?? selectedSource.universe ?? "Unknown"}</dd></div><div><dt>Research IC</dt><dd>{formatMetric(selectedSource.metrics?.oos_ic ?? selectedSource.metrics?.ic)}</dd></div></dl>
              <div className="actions">
                <button type="button" onClick={() => insertOnCanvas(`result:${selectedSource.id}`)}>Use tested snapshot</button>
                <button type="button" className="primary-action" onClick={() => openResultAsEditableCopy(selectedSource)}>Open as editable copy</button>
              </div>
            </section>
          ) : selectedNode ? (
            <section>
              <span className="mode-chip">{selectedData?.origin?.replace("_", " ") ?? selectedData?.kind}</span>
              <h3>{String(selectedData?.label ?? "Block")}</h3>
              <p>{String(selectedData?.description ?? "Calculation block in the current formula.")}</p>
              <p className="hint">Output: {selectedData?.outType}</p>
              {selectedFormulaUpdate && (
                <button type="button" className="primary-action" onClick={() => updateSelectedFormulaReference(selectedFormulaUpdate)}>
                  Update to v{selectedFormulaUpdate.revision ?? 1}
                </button>
              )}
              {selectedData?.kind === "function" && selectedData.inputTypes?.some(isInlineInputType) && (
                <details>
                  <summary>Advanced numeric bindings</summary>
                  <p className="hint">Keep the inline literal, or bind it to a compatible exposed formula input.</p>
                  {selectedData.inputTypes.map((type, index) => {
                    if (!isInlineInputType(type)) return null;
                    const incoming = edges.find((edge) => (
                      edge.target === selectedNode.id && edge.targetHandle === `input-${index}`
                    ));
                    const compatibleInputs = nodes.filter((node) => {
                      const data = node.data as FormulaNodeData;
                      return data.kind === "input" && typesCompatible(data.outType, type);
                    });
                    return (
                      <label className="field" key={`${selectedNode.id}-binding-${index}`}>
                        <span className="field-label">{selectedData.inputNames?.[index] ?? `Input ${index + 1}`} source</span>
                        <select
                          aria-label={`${selectedData.inputNames?.[index] ?? `Input ${index + 1}`} source`}
                          value={incoming?.source ?? ""}
                          onChange={(event) => bindInlineInput(selectedNode.id, index, event.target.value)}
                        >
                          <option value="">Inline {type}</option>
                          {compatibleInputs.map((node) => (
                            <option key={node.id} value={node.id}>${String((node.data as FormulaNodeData).label)}</option>
                          ))}
                        </select>
                      </label>
                    );
                  })}
                </details>
              )}
              {selectedData?.kind === "factor" && <button type="button" className="primary-action" onClick={expandSelectedFactor}>Expand copy</button>}
            </section>
          ) : (
            <section className="formula-definition">
              <h3>Formula definition</h3>
              <label className="field"><span className="field-label">Internal name</span><input value={name} disabled={Boolean(loadedName)} onChange={(event) => { setName(event.target.value); setDirty(true); }} /></label>
              <label className="field"><span className="field-label">Display name</span><input value={displayName} onChange={(event) => { setDisplayName(event.target.value); setDirty(true); }} /></label>
              <label className="field"><span className="field-label">Description</span><textarea rows={4} value={description} onChange={(event) => { setDescription(event.target.value); setDirty(true); }} /></label>
              <label className="field"><span className="field-label">Category</span><input list="formula-categories" value={category} onChange={(event) => { setCategory(event.target.value); setDirty(true); }} /></label>
              <datalist id="formula-categories">{categories.map((item) => <option key={item} value={item} />)}</datalist>
              <label className="field"><span className="field-label">Output type</span><select value={outType} onChange={(event) => updateOutputType(event.target.value)}>{TYPE_OPTIONS.map((type) => <option key={type}>{type}</option>)}</select></label>
              <details className="formula-inputs">
                <summary>Advanced · exposed inputs <span>{inputs.length}</span></summary>
                <p className="hint">Only add an input when another formula must supply it. Most formulas should read Data directly.</p>
                <div className="formula-inputs__head"><span className="field-label">Named inputs</span><button type="button" onClick={addInput}>Add input</button></div>
                {inputs.map((input, index) => (
                  <div className="formula-input-row" key={`input-${index}`}>
                    <input aria-label={`Input ${index + 1} name`} value={input.name} onChange={(event) => updateInput(index, { name: event.target.value })} />
                    <select aria-label={`Input ${index + 1} type`} value={input.type} onChange={(event) => updateInput(index, { type: event.target.value })}>{TYPE_OPTIONS.map((type) => <option key={type}>{type}</option>)}</select>
                    <input aria-label={`Input ${index + 1} description`} value={input.description} onChange={(event) => updateInput(index, { description: event.target.value })} placeholder="Description" />
                    {(input.type === "window" || input.type === "scalar") && <input aria-label={`Input ${index + 1} default`} type="number" step={input.type === "window" ? 1 : "any"} min={input.type === "window" ? 1 : undefined} value={input.default ?? ""} onChange={(event) => updateInput(index, { default: event.target.value === "" ? null : Number(event.target.value) })} placeholder="Default" />}
                    <button type="button" onClick={() => removeInput(index)} aria-label={`Remove input ${index + 1}`}>Remove</button>
                  </div>
                ))}
              </details>
              {formulaDetail && formulaDetail.revisions.length > 1 && <details><summary>History ({formulaDetail.revisions.length})</summary><ul className="formula-history">{formulaDetail.revisions.map((revision) => <li key={revision.runtime_name}>v{revision.revision} <code>{revision.runtime_name}</code></li>)}</ul></details>}
              {loadedName && <button type="button" className="ghost danger" onClick={async () => { if (!window.confirm(`Delete ${displayName}?`)) return; try { await deleteFormula(loadedName); newFormula(); refresh(); } catch (reason) { setError(String(reason)); } }}>Delete formula</button>}
            </section>
          )}
        </aside>
      </div>

      {pendingInsert && (
        <div className="formula-dialog-backdrop" role="presentation">
          <section className="formula-dialog" role="dialog" aria-modal="true" aria-labelledby="insert-title">
            <h3 id="insert-title">This input already has a value</h3>
            <p>Wrap keeps the current calculation inside the new block. Replace disconnects it and leaves the old blocks on the canvas.</p>
            <div className="actions"><button type="button" className="primary-action" onClick={() => resolveOccupied("wrap")}>Wrap existing</button><button type="button" onClick={() => resolveOccupied("replace")}>Replace</button><button type="button" onClick={() => setPendingInsert(null)}>Cancel</button></div>
          </section>
        </div>
      )}

      {pendingSave && (
        <div className="formula-dialog-backdrop" role="presentation">
          <section className="formula-dialog" role="dialog" aria-modal="true" aria-labelledby="save-impact-title">
            <h3 id="save-impact-title">This calculation is already in use</h3>
            <p>Changing it affects {pendingSave.impact.transitive_formulas.length} saved formula(s). {pendingSave.impact.factors.length} formula result(s), {pendingSave.impact.sessions.length} session(s), and {pendingSave.impact.runs?.length ?? 0} active run(s) will remain pinned to their historical calculation.</p>
            {pendingSave.impact.transitive_formulas.length > 0 && <p className="hint">Formula references: {pendingSave.impact.transitive_formulas.join(", ")}</p>}
            <div className="actions"><button type="button" className="primary-action" onClick={() => persist(pendingSave.spec, "upgrade_references").catch((reason) => setError(String(reason)))}>Upgrade formula references</button><button type="button" onClick={branchPending}>Branch as new formula</button><button type="button" onClick={() => setPendingSave(null)}>Cancel</button></div>
          </section>
        </div>
      )}

      <FormulaBacktestDrawer
        open={backtestOpen}
        source={backtestSource}
        inputs={backtestSource?.kind === "draft" ? backtestSource.inputs : inputs}
        formulas={formulas}
        defaultName={`${displayName || name} backtest`}
        defaultUniverse={defaultUniverse}
        onDataSync={onDataSync}
        onClose={() => setBacktestOpen(false)}
      />
    </div>
  );
}
