// P6-T1 (pure): turn a factor tree into React Flow nodes + edges with a tidy layout.
// Pure and deterministic so it is trivially unit-testable (node per primitive, edges = nodes-1).

import type { Edge, Node } from "@xyflow/react";
import type { FactorNode } from "../api/types";

export interface TreeNodeData extends Record<string, unknown> {
  name: string;
  value?: number;
}

export type FlowTree = { nodes: Node<TreeNodeData>[]; edges: Edge[] };

const X_GAP = 150;
const Y_GAP = 90;

export function formatValue(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

export function treeToFlow(root: FactorNode): FlowTree {
  const nodes: Node<TreeNodeData>[] = [];
  const edges: Edge[] = [];
  let nextId = 0;
  let nextLeafX = 0;

  function visit(node: FactorNode, depth: number): { id: string; x: number } {
    const id = `n${nextId++}`;
    const childResults = (node.children ?? []).map((child) => visit(child, depth + 1));
    const x = childResults.length
      ? childResults.reduce((sum, c) => sum + c.x, 0) / childResults.length
      : nextLeafX++ * X_GAP;

    nodes.push({
      id,
      type: "factorNode",
      position: { x, y: depth * Y_GAP },
      data: { name: node.name, value: node.value },
    });
    for (const child of childResults) {
      edges.push({ id: `${id}-${child.id}`, source: id, target: child.id });
    }
    return { id, x };
  }

  visit(root, 0);
  return { nodes, edges };
}
