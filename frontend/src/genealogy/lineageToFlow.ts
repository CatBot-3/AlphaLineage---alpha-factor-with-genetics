// P6-T2 (pure): lay the lineage out as a layered DAG — generation on the y axis, index on x —
// with a parent->child edge carrying the operation. Pure, so navigability is unit-testable.

import type { Edge, Node } from "@xyflow/react";
import type { Lineage } from "../api/types";

export interface LineageNodeData extends Record<string, unknown> {
  label: string;
  generation: number;
  op: string;
}

const X_GAP = 110;
const Y_GAP = 120;

export function lineageToFlow(lineage: Lineage): {
  nodes: Node<LineageNodeData>[];
  edges: Edge[];
} {
  const indexInGen = new Map<number, number>();
  const counts = new Map<number, number>();
  for (const node of lineage.nodes) {
    const i = counts.get(node.generation) ?? 0;
    indexInGen.set(node.id, i);
    counts.set(node.generation, i + 1);
  }

  const nodes: Node<LineageNodeData>[] = lineage.nodes.map((node) => ({
    id: String(node.id),
    type: "lineageNode",
    position: { x: (indexInGen.get(node.id) ?? 0) * X_GAP, y: node.generation * Y_GAP },
    data: { label: `#${node.id}`, generation: node.generation, op: node.op },
  }));

  const edges: Edge[] = [];
  for (const node of lineage.nodes) {
    for (const parent of node.parents) {
      edges.push({
        id: `${parent}-${node.id}`,
        source: String(parent),
        target: String(node.id),
        label: node.op,
      });
    }
  }
  return { nodes, edges };
}
