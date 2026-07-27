import { useEffect, useMemo, useState } from "react";
import { getPrimitives } from "../api/client";
import type { FactorNode, PrimitiveInfo } from "../api/types";
import { FormulaCanvas } from "../extend/FormulaCanvas";
import {
  bodyToFormulaGraph,
  type FormulaNodeData,
} from "../extend/formulaGraph";
import { serializeFormula } from "../extend/formulaText";

function inferredOutputType(factor: FactorNode, primitives: PrimitiveInfo[]): string {
  const primitive = primitives.find((item) => (
    item.name === factor.name ||
    item.runtime_name === factor.name ||
    item.logical_name === factor.name
  ));
  if (primitive?.out_type) return primitive.out_type;
  if (factor.name.startsWith("gt") || factor.name.startsWith("lt") || factor.name.startsWith("eq")) {
    return "bool";
  }
  return "series";
}

function withTreeFallbacks(factor: FactorNode, primitives: PrimitiveInfo[]): PrimitiveInfo[] {
  const known = new Set(primitives.flatMap((item) => [
    item.name,
    item.runtime_name,
    item.logical_name,
  ].filter((name): name is string => Boolean(name))));
  const inferred: PrimitiveInfo[] = [];

  function visit(node: FactorNode) {
    const children = node.children ?? [];
    if (children.length > 0 && !known.has(node.name)) {
      known.add(node.name);
      const argTypes = children.map((child) => (
        child.name === "window" ? "window" : child.name === "const" ? "scalar" : "series"
      ));
      inferred.push({
        name: node.name,
        display_name: node.name.replace(/_/g, " "),
        description: "Pinned operator from the saved formula snapshot.",
        kind: "operator",
        arg_types: argTypes,
        inputs: argTypes.map((type, index) => ({
          name: type === "window"
            ? "lookback"
            : type === "scalar"
              ? "value"
              : children.length === 2
                ? (index === 0 ? "left" : "right")
                : `series ${index + 1}`,
          type,
          description: "Input preserved by the saved expression.",
        })),
        out_type: "series",
        origin: "builtin",
        user: false,
      });
    }
    children.forEach(visit);
  }

  visit(factor);
  return [...primitives, ...inferred];
}

export function ReadOnlyFormulaGraph({
  factor,
  selectedNodeId,
  onSelectNode,
  onInspect,
  compact = false,
  ariaLabel = "Read-only formula diagram",
}: {
  factor: FactorNode;
  selectedNodeId?: string | null;
  onSelectNode?: (id: string | null) => void;
  onInspect?: (data: FormulaNodeData | null) => void;
  compact?: boolean;
  ariaLabel?: string;
}) {
  const [primitives, setPrimitives] = useState<PrimitiveInfo[]>([]);
  const [internalSelection, setInternalSelection] = useState<string | null>(null);
  const expression = useMemo(() => serializeFormula(factor), [factor]);

  useEffect(() => {
    let current = true;
    getPrimitives()
      .then((items) => {
        if (current) setPrimitives(items);
      })
      .catch(() => {
        // A saved result remains inspectable against its embedded tree when the
        // primitive catalog is unavailable (for example, in static demo mode).
      });
    return () => {
      current = false;
    };
  }, []);

  useEffect(() => {
    setInternalSelection(null);
    onInspect?.(null);
  }, [expression, onInspect]);

  const graph = useMemo(
    () => {
      const catalog = withTreeFallbacks(factor, primitives);
      return bodyToFormulaGraph(factor, [], catalog, inferredOutputType(factor, catalog));
    },
    [factor, primitives],
  );
  const activeSelection = selectedNodeId === undefined ? internalSelection : selectedNodeId;

  function select(id: string | null) {
    if (selectedNodeId === undefined) setInternalSelection(id);
    onSelectNode?.(id);
    const node = id ? graph.nodes.find((item) => item.id === id) : undefined;
    onInspect?.(node ? node.data as FormulaNodeData : null);
  }

  return (
    <div
      className={`readonly-formula-graph${compact ? " readonly-formula-graph--compact" : ""}`}
      data-testid="readonly-formula-graph"
    >
      <FormulaCanvas
        key={`${expression}:${primitives.length}`}
        mode="inspect"
        nodes={graph.nodes}
        edges={graph.edges}
        selectedNodeId={activeSelection ?? null}
        onSelectNode={select}
        compact={compact}
        ariaLabel={ariaLabel}
      />
    </div>
  );
}
