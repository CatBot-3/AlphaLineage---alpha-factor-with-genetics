// Pure tree -> graph conversion: the reverse of graphToBody, used to render an existing
// formula body (or freshly parsed text) in the React Flow editor. Pre-order x assignment keeps
// children strictly left-to-right so graphToBody re-reads argument order losslessly.

import type { FactorNode } from "../api/types";
import type { ComposerEdge, ComposerNode } from "./graphToBody";

const X_STEP = 90;
const Y_STEP = 80;

export function bodyToGraph(tree: FactorNode): { nodes: ComposerNode[]; edges: ComposerEdge[] } {
  const nodes: ComposerNode[] = [];
  const edges: ComposerEdge[] = [];
  let counter = 0;

  function visit(node: FactorNode, depth: number, parentId: string | null): void {
    const id = `n${counter}`;
    const order = counter; // pre-order index -> strictly increasing x left to right
    counter += 1;

    const composer: ComposerNode = { id, kind: node.name, x: order * X_STEP, y: depth * Y_STEP };
    if (node.name === "$arg") {
      composer.argIndex = node.value ?? 0;
    } else if (!node.children?.length && node.value !== undefined) {
      composer.value = node.value; // const / window leaf
    }
    nodes.push(composer);
    if (parentId !== null) edges.push({ source: parentId, target: id });

    for (const child of node.children ?? []) visit(child, depth + 1, id);
  }

  visit(tree, 0, null);
  return { nodes, edges };
}
