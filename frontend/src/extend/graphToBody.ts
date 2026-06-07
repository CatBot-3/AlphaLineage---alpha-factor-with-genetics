// P7-T2 (pure): turn the operator composer's node graph into a macro body tree.
// Children are ordered left-to-right by x position (the leftmost child is arg 0 of an op).

import type { FactorNode } from "../api/types";

export interface ComposerNode {
  id: string;
  kind: string; // a primitive name, "$arg", "const", or "window"
  argIndex?: number; // for "$arg"
  value?: number; // for const / window
  x: number; // horizontal position (orders sibling children)
}

export interface ComposerEdge {
  source: string;
  target: string;
}

/** The body's root = the single node that is not a child of any other node. */
export function findRoot(nodes: ComposerNode[], edges: ComposerEdge[]): string | null {
  const targets = new Set(edges.map((e) => e.target));
  const roots = nodes.filter((n) => !targets.has(n.id));
  return roots.length === 1 ? roots[0].id : null;
}

export function graphToBody(
  nodes: ComposerNode[],
  edges: ComposerEdge[],
  rootId: string,
): FactorNode {
  const byId = new Map(nodes.map((n) => [n.id, n]));

  function build(id: string): FactorNode {
    const node = byId.get(id);
    if (!node) throw new Error(`unknown node ${id}`);
    if (node.kind === "$arg") {
      return { name: "$arg", value: node.argIndex ?? 0 };
    }
    if (node.value !== undefined) {
      return { name: node.kind, value: node.value };
    }
    const children = edges
      .filter((e) => e.source === id)
      .map((e) => byId.get(e.target))
      .filter((c): c is ComposerNode => c !== undefined)
      .sort((a, b) => a.x - b.x)
      .map((c) => build(c.id));
    return children.length ? { name: node.kind, children } : { name: node.kind };
  }

  return build(rootId);
}
