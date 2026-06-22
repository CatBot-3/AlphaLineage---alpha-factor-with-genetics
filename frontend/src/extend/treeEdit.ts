// Pure immutable edits over a formula body tree, used by the params pane of the editor.

import type { FactorNode } from "../api/types";

export interface LeafRef {
  path: number[];
  node: FactorNode;
}

/** Every editable numeric/argument leaf (window, const, $arg) with its path from the root. */
export function paramLeaves(tree: FactorNode): LeafRef[] {
  const out: LeafRef[] = [];
  function walk(node: FactorNode, path: number[]): void {
    const isLeaf = !node.children?.length;
    if (isLeaf && (node.name === "window" || node.name === "const" || node.name === "$arg")) {
      out.push({ path, node });
    }
    node.children?.forEach((c, i) => walk(c, [...path, i]));
  }
  walk(tree, []);
  return out;
}

export function replaceAt(tree: FactorNode, path: number[], next: FactorNode): FactorNode {
  if (path.length === 0) return next;
  const [head, ...rest] = path;
  const children = (tree.children ?? []).map((c, i) =>
    i === head ? replaceAt(c, rest, next) : c,
  );
  return { ...tree, children };
}

/** Set the numeric value of a const/window leaf at ``path``. */
export function setLeafValue(tree: FactorNode, path: number[], value: number): FactorNode {
  const leaf = nodeAt(tree, path);
  if (!leaf) return tree;
  return replaceAt(tree, path, { ...leaf, value });
}

export function nodeAt(tree: FactorNode, path: number[]): FactorNode | null {
  let cur: FactorNode | undefined = tree;
  for (const i of path) cur = cur?.children?.[i];
  return cur ?? null;
}

/**
 * Replace a fixed window/const leaf with the ``$arg`` placeholder ``argIndex``, exposing it as a
 * tunable argument (the GP mutates argument values at call sites). Returns the new tree.
 */
export function promoteToArg(tree: FactorNode, path: number[], argIndex: number): FactorNode {
  return replaceAt(tree, path, { name: "$arg", value: argIndex });
}

/** The DSL type a const/window/$arg leaf supplies (window | scalar | the declared arg type). */
export function leafType(node: FactorNode, argTypes: string[]): string {
  if (node.name === "window") return "window";
  if (node.name === "const") return "scalar";
  if (node.name === "$arg") return argTypes[node.value ?? 0] ?? "series";
  return "series";
}
