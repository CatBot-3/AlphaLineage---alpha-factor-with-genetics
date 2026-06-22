import type { Connection, ReactFlowInstance } from "@xyflow/react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  addFormula,
  deleteFormula,
  getCategories,
  getFormula,
  getFormulaImpact,
  getPrimitives,
  listFactors,
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
  FormulaSpec,
  PrimitiveInfo,
  SavedFactor,
} from "../api/types";
import { FormulaCanvas } from "./FormulaCanvas";
import {
  autoLayoutGraph,
  blankFormulaGraph,
  bodyToFormulaGraph,
  graphToFormulaBody,
  nodeOutputType,
  OUTPUT_NODE_ID,
  primitiveNode,
  targetInputType,
  typesCompatible,
  type FormulaGraph,
  type FormulaNodeData,
} from "./formulaGraph";
import { parseFormula, serializeFormula } from "./formulaText";

const TYPE_OPTIONS = ["series", "signal", "window", "scalar", "bool"];
const DEFAULT_INPUTS: FormulaInputSpec[] = [
  { name: "price", type: "series", description: "Price or derived series to transform." },
];

const DEMO_PRIMITIVES: PrimitiveInfo[] = [
  { name: "close", display_name: "Close", description: "Closing price for each symbol and date.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
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

function factorOutputType(factor: SavedFactor, primitives: PrimitiveInfo[]): string {
  const root = factor.expanded_tree ?? factor.tree;
  return primitives.find((primitive) => primitive.name === root.name)?.out_type ?? "signal";
}

export function FormulaEditorPage({
  formulaDraft,
  onFormulaDraftChange,
  canSubmit = true,
}: {
  formulaDraft?: FormulaDraft;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
  canSubmit?: boolean;
}) {
  const initialInputs = formulaDraft?.inputs?.length ? formulaDraft.inputs : DEFAULT_INPUTS;
  const initialOutType = formulaDraft?.out_type ?? "signal";
  const initialGraph = draftGraph(formulaDraft, initialInputs, initialOutType);

  const [primitives, setPrimitives] = useState<PrimitiveInfo[]>(canSubmit ? [] : DEMO_PRIMITIVES);
  const [formulas, setFormulas] = useState<FormulaSpec[]>([]);
  const [factors, setFactors] = useState<SavedFactor[]>([]);
  const [categories, setCategories] = useState<string[]>(["custom"]);
  const [name, setName] = useState(formulaDraft?.name ?? "my_formula");
  const [displayName, setDisplayName] = useState(formulaDraft?.display_name ?? "My formula");
  const [description, setDescription] = useState(formulaDraft?.description ?? "");
  const [inputs, setInputs] = useState<FormulaInputSpec[]>(initialInputs);
  const [outType, setOutType] = useState(initialOutType);
  const [category, setCategory] = useState(formulaDraft?.category ?? "custom");
  const [nodes, setNodes] = useState<FormulaDraftNode[]>(initialGraph.nodes);
  const [edges, setEdges] = useState<FormulaDraftEdge[]>(initialGraph.edges);
  const [expression, setExpression] = useState(formulaDraft?.expression ?? "");
  const [activeMode, setActiveMode] = useState<"visual" | "expression">(formulaDraft?.activeMode ?? "visual");
  const [mobilePane, setMobilePane] = useState<"library" | "canvas" | "inspector">("canvas");
  const [loadedName, setLoadedName] = useState<string | null>(formulaDraft?.loadedName ?? null);
  const [loadedRevision, setLoadedRevision] = useState<number | null>(formulaDraft?.loadedRevision ?? null);
  const [formulaDetail, setFormulaDetail] = useState<FormulaDetail | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(formulaDraft?.selectedNodeId ?? null);
  const [selectedSlot, setSelectedSlot] = useState<{ nodeId: string; index: number } | null>(null);
  const [selectedSource, setSelectedSource] = useState<PrimitiveInfo | SavedFactor | null>(null);
  const [search, setSearch] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [pendingInsert, setPendingInsert] = useState<PendingInsert | null>(null);
  const [pendingSave, setPendingSave] = useState<{ spec: FormulaSpec; impact: FormulaImpact } | null>(null);
  const [past, setPast] = useState<FormulaGraph[]>([]);
  const [future, setFuture] = useState<FormulaGraph[]>([]);
  const flowRef = useRef<ReactFlowInstance | null>(null);
  const nodeCounter = useRef(nodes.length + 1);
  const shouldSeedGraph = useRef(!formulaDraft?.graphNodes?.length);
  const seedBody = useRef(formulaDraft?.body);

  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedData = selectedNode?.data as FormulaNodeData | undefined;

  function refresh() {
    if (!canSubmit) return;
    Promise.all([getPrimitives(), listFormulas(), listFactors(), getCategories()])
      .then(([nextPrimitives, nextFormulas, nextFactors, nextCategories]) => {
        setPrimitives(nextPrimitives);
        setFormulas(nextFormulas);
        setFactors(nextFactors);
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
        return;
      }
      const rank = primitives.find((primitive) => primitive.logical_name === "rank" || primitive.name === "rank");
      if (rank) {
        const body: FactorNode = { name: rank.name, children: [{ name: "$arg", value: 0 }] };
        const graph = bodyToFormulaGraph(body, inputs, primitives, outType);
        setNodes(graph.nodes);
        setEdges(graph.edges);
        setExpression(serializeFormula(body, inputs));
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
    const warn = (event: BeforeUnloadEvent) => {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  function graphSnapshot(): FormulaGraph {
    return { nodes, edges };
  }

  function commitGraph(next: FormulaGraph) {
    setPast((history) => [...history.slice(-39), graphSnapshot()]);
    setFuture([]);
    setNodes(next.nodes);
    setEdges(next.edges);
    setDirty(true);
    try {
      setExpression(serializeFormula(graphToFormulaBody(next.nodes, next.edges), inputs));
      setParseError(null);
    } catch {
      // Incomplete visual drafts are expected while blocks are being assembled.
    }
  }

  function undo() {
    const previous = past[past.length - 1];
    if (!previous) return;
    setFuture((items) => [graphSnapshot(), ...items]);
    setPast((items) => items.slice(0, -1));
    setNodes(previous.nodes);
    setEdges(previous.edges);
  }

  function redo() {
    const next = future[0];
    if (!next) return;
    setPast((items) => [...items, graphSnapshot()]);
    setFuture((items) => items.slice(1));
    setNodes(next.nodes);
    setEdges(next.edges);
  }

  function updateOutputType(value: string) {
    setOutType(value);
    setNodes((items) => items.map((node) => node.id === OUTPUT_NODE_ID
      ? { ...node, data: { ...node.data, outType: value } }
      : node));
    setDirty(true);
  }

  function updateInput(index: number, patch: Partial<FormulaInputSpec>) {
    setInputs((items) => items.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item));
    setNodes((items) => items.map((node) => node.id === `formula-input-${index}`
      ? { ...node, data: { ...node.data, label: patch.name ?? node.data.label, outType: patch.type ?? node.data.outType, description: patch.description ?? node.data.description } }
      : node));
    setDirty(true);
  }

  function addInput() {
    const index = inputs.length;
    const input = { name: `input_${index + 1}`, type: "series", description: "Formula input." };
    setInputs((items) => [...items, input]);
    commitGraph({
      nodes: [...nodes, {
        id: `formula-input-${index}`,
        type: "formula",
        x: 40,
        y: 70 + index * 110,
        data: { kind: "input", label: input.name, description: input.description, outType: input.type, argIndex: index, locked: true },
      }],
      edges,
    });
  }

  function removeInput(index: number) {
    const id = `formula-input-${index}`;
    if (edges.some((edge) => edge.source === id)) {
      setError("Disconnect this input before removing it.");
      return;
    }
    setInputs((items) => items.filter((_, itemIndex) => itemIndex !== index));
    const nextNodes = nodes
      .filter((node) => node.id !== id)
      .map((node) => {
        const data = node.data as FormulaNodeData;
        if (data.kind !== "input" || Number(data.argIndex) < index) return node;
        const nextIndex = Number(data.argIndex) - 1;
        return { ...node, id: `formula-input-${nextIndex}`, data: { ...data, argIndex: nextIndex } };
      });
    commitGraph({ nodes: nextNodes, edges: edges.filter((edge) => edge.source !== id) });
  }

  function nextNodeFor(source: PrimitiveInfo | SavedFactor, point = { x: 260, y: 180 }): FormulaDraftNode {
    const id = `formula-node-${nodeCounter.current++}`;
    if ("kind" in source) return primitiveNode(source, id, point.x, point.y);
    return {
      id,
      type: "formula",
      x: point.x,
      y: point.y,
      data: {
        kind: "factor",
        label: source.name,
        description: source.notes || "Saved alpha factor snapshot.",
        origin: "saved_factor",
        outType: factorOutputType(source, primitives),
        factorId: source.id,
        factorBody: source.expanded_tree ?? source.tree,
        locked: true,
      } satisfies FormulaNodeData,
    };
  }

  function findSource(key: string): PrimitiveInfo | SavedFactor | undefined {
    if (key.startsWith("factor:")) return factors.find((factor) => factor.id === key.slice(7));
    return primitives.find((primitive) => primitive.name === key || primitive.logical_name === key);
  }

  function insertOnCanvas(key: string, point?: { x: number; y: number }) {
    const source = findSource(key);
    if (!source) return;
    if (selectedSlot) {
      insertIntoSlot(selectedSlot.nodeId, selectedSlot.index, key);
      return;
    }
    const node = nextNodeFor(source, point);
    commitGraph({ nodes: [...nodes, node], edges });
    setSelectedNodeId(node.id);
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

  function duplicateSelected() {
    if (!selectedNode || selectedNode.id === OUTPUT_NODE_ID || (selectedData?.kind === "input")) return;
    const copy = { ...selectedNode, id: `formula-node-${nodeCounter.current++}`, x: selectedNode.x + 35, y: selectedNode.y + 35, data: { ...selectedNode.data } };
    commitGraph({ nodes: [...nodes, copy], edges });
    setSelectedNodeId(copy.id);
  }

  function deleteSelected() {
    if (!selectedNode || selectedNode.id === OUTPUT_NODE_ID || selectedData?.kind === "input") return;
    commitGraph({
      nodes: nodes.filter((node) => node.id !== selectedNode.id),
      edges: edges.filter((edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id),
    });
    setSelectedNodeId(null);
  }

  function expandSelectedFactor() {
    if (!selectedNode || selectedData?.kind !== "factor" || !selectedData.factorBody) return;
    const subgraph = bodyToFormulaGraph(selectedData.factorBody, [], primitives, selectedData.outType);
    const outputEdge = subgraph.edges.find((edge) => edge.target === OUTPUT_NODE_ID);
    if (!outputEdge) return;
    const prefix = `${selectedNode.id}-expanded-`;
    const subNodes = subgraph.nodes
      .filter((node) => node.id !== OUTPUT_NODE_ID)
      .map((node) => ({ ...node, id: `${prefix}${node.id}`, x: node.x + selectedNode.x - 160, y: node.y + selectedNode.y - 80 }));
    const idMap = new Map(subgraph.nodes.filter((node) => node.id !== OUTPUT_NODE_ID).map((node) => [node.id, `${prefix}${node.id}`]));
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
    setDirty(false);
    setMessage(null);
    setError(null);
    if (canSubmit) getFormula(spec.name).then(setFormulaDetail).catch(() => setFormulaDetail(null));
  }

  function newFormula() {
    if (dirty && !window.confirm("Discard the current unsaved draft?")) return;
    const graph = blankFormulaGraph(DEFAULT_INPUTS, "signal");
    setName("my_formula");
    setDisplayName("My formula");
    setDescription("");
    setInputs(DEFAULT_INPUTS);
    setOutType("signal");
    setCategory("custom");
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setExpression("");
    setLoadedName(null);
    setLoadedRevision(null);
    setFormulaDetail(null);
    setDirty(false);
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
    setSelectedSource(null);
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
  }

  const filteredPrimitives = useMemo(() => {
    const query = search.trim().toLowerCase();
    return primitives.filter((primitive) => !query || [primitive.name, primitive.display_name, primitive.description, primitive.category].some((value) => value?.toLowerCase().includes(query)));
  }, [primitives, search]);

  const grouped = useMemo(() => {
    const groups = new Map<string, PrimitiveInfo[]>();
    for (const primitive of filteredPrimitives) {
      const key = primitive.category ?? "uncategorized";
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(primitive);
    }
    return groups;
  }, [filteredPrimitives]);

  return (
    <div className="formula-workspace" data-testid="formula-editor-page">
      <header className="formula-workspace__toolbar">
        <div>
          <h3>Formula workspace</h3>
          <p className="panel-note">Build a reusable calculation from typed inputs and functions.</p>
        </div>
        <div className="formula-toolbar__actions">
          <button type="button" onClick={newFormula}>New</button>
          <button type="button" onClick={save} className="primary-action" disabled={!canSubmit}>Save</button>
          <button type="button" onClick={undo} disabled={!past.length} title="Undo">Undo</button>
          <button type="button" onClick={redo} disabled={!future.length} title="Redo">Redo</button>
          <button type="button" onClick={duplicateSelected} disabled={!selectedNodeId}>Duplicate</button>
          <button type="button" onClick={deleteSelected} disabled={!selectedNodeId}>Delete</button>
          <button type="button" onClick={() => commitGraph(autoLayoutGraph(nodes, edges))}>Auto-layout</button>
          <button type="button" onClick={() => flowRef.current?.fitView({ padding: 0.18 })}>Fit</button>
        </div>
        <div className="formula-mode" role="tablist" aria-label="Formula editor mode">
          <button type="button" role="tab" aria-selected={activeMode === "visual"} onClick={() => setActiveMode("visual")}>Visual</button>
          <button type="button" role="tab" aria-selected={activeMode === "expression"} onClick={() => setActiveMode("expression")}>Expression</button>
        </div>
      </header>

      {!canSubmit && <p className="surface-message">Static demo mode keeps this workspace as a local draft. Connect the backend to validate and save formulas.</p>}
      {message && <p className="ok">{message}</p>}
      {error && <p className="error">{error}</p>}

      <nav className="formula-mobile-nav" aria-label="Formula workspace panels">
        {(["library", "canvas", "inspector"] as const).map((pane) => (
          <button key={pane} type="button" aria-pressed={mobilePane === pane} onClick={() => setMobilePane(pane)}>{pane}</button>
        ))}
      </nav>

      <div className={`formula-workspace__grid formula-pane--${mobilePane}`}>
        <aside className="formula-library" data-testid="formula-library">
          <label className="field">
            <span className="field-label">Find a building block</span>
            <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search functions and fields" />
          </label>
          {[...grouped.entries()].map(([group, items]) => (
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
                    <button type="button" className="ghost" onClick={() => setSelectedSource(item)}>{item.user ? "Open" : "Inspect"}</button>
                  </article>
                ))}
              </div>
            </details>
          ))}

          {factors.length > 0 && (
            <details open>
              <summary>Saved factors <span>{factors.length}</span></summary>
              <div className="formula-library__items">
                {factors.map((factor) => (
                  <article
                    key={factor.id}
                    className="formula-library__item formula-library__item--factor"
                    draggable
                    onDragStart={(event) => event.dataTransfer.setData("application/x-alphalineage-primitive", `factor:${factor.id}`)}
                  >
                    <button type="button" className="formula-library__insert" onClick={() => insertOnCanvas(`factor:${factor.id}`)}>
                      <strong>{factor.name}</strong>
                      <span>{factor.notes || `Saved from ${factor.provenance?.universe ?? "a training session"}.`}</span>
                      <code>Snapshot {"->"} {factorOutputType(factor, primitives)}</code>
                    </button>
                    <button type="button" className="ghost" onClick={() => setSelectedSource(factor)}>Inspect</button>
                  </article>
                ))}
              </div>
            </details>
          )}
        </aside>

        <main className="formula-stage">
          {activeMode === "visual" ? (
            <FormulaCanvas
              nodes={nodes}
              edges={edges}
              selectedNodeId={selectedNodeId}
              selectedSlot={selectedSlot}
              onNodesChange={setNodes}
              onEdgesChange={(nextEdges) => commitGraph({ nodes, edges: nextEdges })}
              onConnect={connect}
              onSelectNode={(id) => { setSelectedNodeId(id); setSelectedSource(null); }}
              onSelectSlot={(nodeId, index) => setSelectedSlot({ nodeId, index })}
              onDropPrimitive={insertIntoSlot}
              onDropCanvas={insertOnCanvas}
              onValueChange={(nodeId, value) => commitGraph({ nodes: nodes.map((node) => node.id === nodeId ? { ...node, data: { ...node.data, value } } : node), edges })}
              onInit={(instance) => { flowRef.current = instance; }}
            />
          ) : (
            <section className="formula-expression-panel">
              <label className="field">
                <span className="field-label">Expression</span>
                <textarea rows={14} value={expression} onChange={(event) => editExpression(event.target.value)} aria-label="Formula expression" spellCheck={false} />
              </label>
              <p className="hint">Named inputs use a dollar sign, for example <code>ts_mean($price, $lookback)</code>.</p>
              {parseError && <p className="error">{parseError}</p>}
            </section>
          )}
        </main>

        <aside className="formula-inspector" data-testid="formula-inspector">
          {selectedSource && "kind" in selectedSource ? (
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
              <span className="mode-chip">Saved factor</span>
              <h3>{selectedSource.name}</h3>
              <p>{selectedSource.notes || "Locked calculation snapshot from training."}</p>
              <dl className="formula-inspector__signature"><div><dt>Saved</dt><dd>{selectedSource.saved_at}</dd></div><div><dt>Universe</dt><dd>{selectedSource.provenance?.universe ?? "Unknown"}</dd></div><div><dt>Research IC</dt><dd>{selectedSource.metrics?.oos_ic?.toFixed?.(3) ?? "-"}</dd></div></dl>
              <button type="button" onClick={() => insertOnCanvas(`factor:${selectedSource.id}`)}>Insert snapshot</button>
            </section>
          ) : selectedNode ? (
            <section>
              <span className="mode-chip">{selectedData?.origin?.replace("_", " ") ?? selectedData?.kind}</span>
              <h3>{String(selectedData?.label ?? "Block")}</h3>
              <p>{String(selectedData?.description ?? "Calculation block in the current formula.")}</p>
              <p className="hint">Output: {selectedData?.outType}</p>
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
              <div className="formula-inputs">
                <div className="formula-inputs__head"><span className="field-label">Named inputs</span><button type="button" onClick={addInput}>Add input</button></div>
                {inputs.map((input, index) => (
                  <div className="formula-input-row" key={`input-${index}`}>
                    <input aria-label={`Input ${index + 1} name`} value={input.name} onChange={(event) => updateInput(index, { name: event.target.value })} />
                    <select aria-label={`Input ${index + 1} type`} value={input.type} onChange={(event) => updateInput(index, { type: event.target.value })}>{TYPE_OPTIONS.map((type) => <option key={type}>{type}</option>)}</select>
                    <input aria-label={`Input ${index + 1} description`} value={input.description} onChange={(event) => updateInput(index, { description: event.target.value })} placeholder="Description" />
                    <button type="button" onClick={() => removeInput(index)} aria-label={`Remove input ${index + 1}`}>Remove</button>
                  </div>
                ))}
              </div>
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
            <p>Changing it affects {pendingSave.impact.transitive_formulas.length} saved formula(s). {pendingSave.impact.factors.length} factor(s) and {pendingSave.impact.sessions.length} session(s) will remain pinned to their historical calculation.</p>
            {pendingSave.impact.transitive_formulas.length > 0 && <p className="hint">Formula references: {pendingSave.impact.transitive_formulas.join(", ")}</p>}
            <div className="actions"><button type="button" className="primary-action" onClick={() => persist(pendingSave.spec, "upgrade_references").catch((reason) => setError(String(reason)))}>Upgrade formula references</button><button type="button" onClick={branchPending}>Branch as new formula</button><button type="button" onClick={() => setPendingSave(null)}>Cancel</button></div>
          </section>
        </div>
      )}
    </div>
  );
}
