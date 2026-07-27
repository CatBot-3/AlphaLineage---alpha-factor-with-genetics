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

/** Collapse purely linear elite-retention runs while preserving real branching ancestry. */
export function compressEliteChains(lineage: Lineage): Lineage {
  const nodes = lineage?.nodes ?? [];
  const children = new Map<number, typeof nodes>();
  for (const node of nodes) {
    for (const parent of node.parents) {
      const current = children.get(parent) ?? [];
      current.push(node);
      children.set(parent, current);
    }
  }
  const removed = new Set<number>();
  const replacements = new Map<number, (typeof nodes)[number]>();

  for (const start of nodes) {
    if (start.op === "elite") continue;
    const chain: typeof nodes = [];
    let cursor = start;
    while (true) {
      const next = (children.get(cursor.id) ?? []).filter((node) => !removed.has(node.id));
      if (next.length !== 1 || next[0].op !== "elite" || next[0].parents.length !== 1) break;
      chain.push(next[0]);
      cursor = next[0];
    }
    if (chain.length < 2) continue;
    for (const intermediate of chain.slice(0, -1)) removed.add(intermediate.id);
    const retained = chain[chain.length - 1];
    replacements.set(retained.id, {
      ...retained,
      parents: [start.id],
      op: `elite × ${chain.length}`,
    });
  }

  return {
    ...lineage,
    nodes: nodes
      .filter((node) => !removed.has(node.id))
      .map((node) => replacements.get(node.id) ?? node),
  };
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
