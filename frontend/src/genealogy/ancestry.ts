// B4 (pure): the ancestor closure of a node - the handful of nodes that actually produced a
// chosen factor. Feeding this filtered lineage to lineageToFlow gives the focused "trace
// ancestry" DAG, which is small enough for a graph to be the right tool.

import type { Lineage } from "../api/types";

export function ancestorClosure(lineage: Lineage, id: number): Lineage {
  const nodes = lineage?.nodes ?? [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const keep = new Set<number>();
  const stack = [id];
  while (stack.length) {
    const current = stack.pop() as number;
    if (keep.has(current)) continue;
    keep.add(current);
    const node = byId.get(current);
    if (node) for (const parent of node.parents) stack.push(parent);
  }
  return { ...lineage, nodes: nodes.filter((n) => keep.has(n.id)) };
}

export function bestFinalNode(lineage: Lineage | null | undefined): number | null {
  const nodes = lineage?.nodes ?? [];
  if (nodes.length === 0) return null;
  const maxGen = Math.max(...nodes.map((n) => n.generation));
  const finals = nodes.filter((n) => n.generation === maxGen);
  let best = finals[0];
  for (const node of finals) {
    if ((node.fitness ?? Number.NEGATIVE_INFINITY) > (best.fitness ?? Number.NEGATIVE_INFINITY)) {
      best = node;
    }
  }
  return best.id;
}
