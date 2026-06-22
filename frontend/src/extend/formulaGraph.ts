import type {
  FactorNode,
  FormulaDraftEdge,
  FormulaDraftNode,
  FormulaInputSpec,
  PrimitiveInfo,
} from "../api/types";

export type FormulaNodeKind = "input" | "function" | "data" | "value" | "factor" | "output";

export interface FormulaNodeData extends Record<string, unknown> {
  kind: FormulaNodeKind;
  label: string;
  description?: string;
  primitiveName?: string;
  logicalName?: string;
  origin?: string;
  inputTypes?: string[];
  inputNames?: string[];
  inputDescriptions?: string[];
  outType: string;
  argIndex?: number;
  value?: number;
  valueKind?: "const" | "window";
  revision?: number;
  factorId?: string;
  factorBody?: FactorNode;
  locked?: boolean;
}

export interface FormulaGraph {
  nodes: FormulaDraftNode[];
  edges: FormulaDraftEdge[];
}

export const OUTPUT_NODE_ID = "formula-output";

function dataOf(node: FormulaDraftNode): FormulaNodeData {
  return node.data as FormulaNodeData;
}

function edgeId(source: string, target: string, handle: string): string {
  return `${source}-${target}-${handle}`;
}

export function typesCompatible(actual: string, expected: string): boolean {
  if (actual === expected) return true;
  return (actual === "series" || actual === "signal") &&
    (expected === "series" || expected === "signal");
}

export function nodeOutputType(node: FormulaDraftNode): string {
  return dataOf(node).outType;
}

export function targetInputType(node: FormulaDraftNode, handle: string | null | undefined): string | null {
  const data = dataOf(node);
  if (data.kind === "output") return data.outType;
  if (!handle?.startsWith("input-")) return null;
  const index = Number(handle.slice("input-".length));
  return data.inputTypes?.[index] ?? null;
}

export function primitiveNode(
  primitive: PrimitiveInfo,
  id: string,
  x: number,
  y: number,
): FormulaDraftNode {
  const inputs = primitive.inputs ?? primitive.arg_types.map((type, index) => ({
    name: `input_${index + 1}`,
    type,
    description: "Function input.",
  }));
  const kind: FormulaNodeKind = primitive.kind === "operand"
    ? "data"
    : primitive.kind === "ephemeral"
      ? "value"
      : "function";
  return {
    id,
    type: "formula",
    x,
    y,
    data: {
      kind,
      label: primitive.display_name ?? primitive.logical_name ?? primitive.name,
      description: primitive.description ?? "Typed calculation block.",
      primitiveName: primitive.runtime_name ?? primitive.name,
      logicalName: primitive.logical_name ?? primitive.name,
      origin: primitive.origin ?? (primitive.user ? "user_formula" : "builtin"),
      inputTypes: inputs.map((item) => item.type),
      inputNames: inputs.map((item) => item.name),
      inputDescriptions: inputs.map((item) => item.description),
      outType: primitive.out_type,
      revision: primitive.revision ?? undefined,
      value: primitive.name === "window" ? 5 : primitive.name === "const" ? 1 : undefined,
      valueKind: primitive.name === "window" ? "window" : primitive.name === "const" ? "const" : undefined,
      locked: primitive.origin === "builtin" || primitive.origin === "data",
    } satisfies FormulaNodeData,
  };
}

export function blankFormulaGraph(inputs: FormulaInputSpec[], outType = "signal"): FormulaGraph {
  const nodes: FormulaDraftNode[] = inputs.map((input, index) => ({
    id: `formula-input-${index}`,
    type: "formula",
    x: 40,
    y: 70 + index * 110,
    data: {
      kind: "input",
      label: input.name,
      description: input.description || "Formula input.",
      outType: input.type,
      argIndex: index,
      locked: true,
    } satisfies FormulaNodeData,
  }));
  nodes.push({
    id: OUTPUT_NODE_ID,
    type: "formula",
    x: 700,
    y: 160,
    data: {
      kind: "output",
      label: "Formula output",
      description: "The value returned by this formula.",
      outType,
      locked: true,
    } satisfies FormulaNodeData,
  });
  return { nodes, edges: [] };
}

export function bodyToFormulaGraph(
  body: FactorNode,
  inputs: FormulaInputSpec[],
  primitives: PrimitiveInfo[],
  outType: string,
): FormulaGraph {
  const graph = blankFormulaGraph(inputs, outType);
  const byName = new Map(primitives.map((primitive) => [primitive.name, primitive]));
  let counter = 0;

  function visit(tree: FactorNode, depth: number, row: number): string {
    if (tree.name === "$arg") {
      const index = Number(tree.value ?? 0);
      return `formula-input-${index}`;
    }
    const id = `formula-node-${counter++}`;
    const primitive = byName.get(tree.name);
    let node: FormulaDraftNode;
    if (primitive) {
      node = primitiveNode(primitive, id, 160 + depth * 210, 60 + row * 105);
    } else {
      node = {
        id,
        type: "formula",
        x: 160 + depth * 210,
        y: 60 + row * 105,
        data: {
          kind: tree.value !== undefined ? "value" : "data",
          label: tree.name,
          primitiveName: tree.name,
          outType: tree.name === "window" ? "window" : tree.name === "const" ? "scalar" : "series",
          value: tree.value,
          valueKind: tree.name === "window" ? "window" : tree.name === "const" ? "const" : undefined,
        } satisfies FormulaNodeData,
      };
    }
    if (tree.value !== undefined) node.data = { ...node.data, value: tree.value };
    graph.nodes.push(node);
    for (const [index, child] of (tree.children ?? []).entries()) {
      const childId = visit(child, Math.max(0, depth - 1), row + index);
      graph.edges.push({
        id: edgeId(childId, id, `input-${index}`),
        source: childId,
        target: id,
        sourceHandle: "output",
        targetHandle: `input-${index}`,
      });
    }
    return id;
  }

  const root = visit(body, 2, 0);
  graph.edges.push({
    id: edgeId(root, OUTPUT_NODE_ID, "result"),
    source: root,
    target: OUTPUT_NODE_ID,
    sourceHandle: "output",
    targetHandle: "result",
  });
  return autoLayoutGraph(graph.nodes, graph.edges);
}

export function graphToFormulaBody(
  nodes: FormulaDraftNode[],
  edges: FormulaDraftEdge[],
): FactorNode {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const outputEdge = edges.find((edge) => edge.target === OUTPUT_NODE_ID && edge.targetHandle === "result");
  if (!outputEdge) throw new Error("Connect a calculation to Formula output.");
  const stack = new Set<string>();

  function build(id: string): FactorNode {
    if (stack.has(id)) throw new Error("Formula connections cannot contain a cycle.");
    const node = byId.get(id);
    if (!node) throw new Error(`Unknown formula node ${id}.`);
    const data = dataOf(node);
    if (data.kind === "input") return { name: "$arg", value: data.argIndex ?? 0 };
    if (data.kind === "factor") {
      if (!data.factorBody) throw new Error(`${data.label} has no saved calculation snapshot.`);
      return data.factorBody;
    }
    if (data.kind === "value") {
      return { name: data.valueKind ?? data.primitiveName ?? "const", value: Number(data.value ?? 0) };
    }
    if (data.kind === "data") return { name: data.primitiveName ?? String(data.label) };
    if (data.kind !== "function") throw new Error(`${data.label} cannot feed another block.`);

    stack.add(id);
    const children = (data.inputTypes ?? []).map((_, index) => {
      const incoming = edges.find((edge) => edge.target === id && edge.targetHandle === `input-${index}`);
      if (!incoming) throw new Error(`${data.label}: connect ${data.inputNames?.[index] ?? `input ${index + 1}`}.`);
      return build(incoming.source);
    });
    stack.delete(id);
    return { name: data.primitiveName ?? String(data.label), children };
  }

  return build(outputEdge.source);
}

export function connectionCreatesCycle(
  edges: FormulaDraftEdge[],
  source: string,
  target: string,
): boolean {
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    const list = outgoing.get(edge.source) ?? [];
    list.push(edge.target);
    outgoing.set(edge.source, list);
  }
  const pending = [target];
  const visited = new Set<string>();
  while (pending.length) {
    const current = pending.pop()!;
    if (current === source) return true;
    if (visited.has(current)) continue;
    visited.add(current);
    pending.push(...(outgoing.get(current) ?? []));
  }
  return false;
}

export function autoLayoutGraph(
  nodes: FormulaDraftNode[],
  edges: FormulaDraftEdge[],
): FormulaGraph {
  const depth = new Map<string, number>([[OUTPUT_NODE_ID, 0]]);
  const pending = [OUTPUT_NODE_ID];
  while (pending.length) {
    const target = pending.shift()!;
    const nextDepth = (depth.get(target) ?? 0) + 1;
    for (const edge of edges.filter((item) => item.target === target)) {
      if ((depth.get(edge.source) ?? -1) < nextDepth) depth.set(edge.source, nextDepth);
      pending.push(edge.source);
    }
  }
  const maxDepth = Math.max(1, ...depth.values());
  const rowByDepth = new Map<number, number>();
  return {
    nodes: nodes.map((node) => {
      const nodeDepth = depth.get(node.id) ?? maxDepth;
      const row = rowByDepth.get(nodeDepth) ?? 0;
      rowByDepth.set(nodeDepth, row + 1);
      return { ...node, x: 40 + (maxDepth - nodeDepth) * 235, y: 55 + row * 125 };
    }),
    edges,
  };
}
