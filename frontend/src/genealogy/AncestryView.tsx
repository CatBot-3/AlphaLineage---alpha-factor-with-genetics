// B4 Design B (trace ancestry): render only a chosen node's ancestor closure as a React Flow
// DAG. The closure is tens of nodes at most, so a graph genuinely beats a list here - it answers
// "where did this factor come from?" spatially.

import { Background, ReactFlow, type ReactFlowInstance } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Lineage } from "../api/types";
import { serializeFormula } from "../extend/formulaText";
import { ancestorClosure, compressEliteChains } from "./ancestry";
import { LineageNodeCard } from "./LineageNodeCard";
import { lineageToFlow } from "./lineageToFlow";

const NODE_TYPES = { lineageNode: LineageNodeCard };

export function lineageKeyboardNeighbor(
  lineage: Lineage,
  currentId: number,
  direction: "up" | "down" | "left" | "right",
): number {
  const current = lineage.nodes.find((node) => node.id === currentId);
  if (!current) return lineage.nodes[0]?.id ?? currentId;
  if (direction === "up") {
    return current.parents.find((id) => lineage.nodes.some((node) => node.id === id)) ?? currentId;
  }
  if (direction === "down") {
    return lineage.nodes
      .filter((node) => node.parents.includes(currentId))
      .sort((left, right) => left.id - right.id)[0]?.id ?? currentId;
  }
  const peers = lineage.nodes
    .filter((node) => node.generation === current.generation)
    .sort((left, right) => left.id - right.id);
  const index = peers.findIndex((node) => node.id === currentId);
  const offset = direction === "left" ? -1 : 1;
  return peers[index + offset]?.id ?? currentId;
}

export function AncestryView({
  lineage,
  focusId,
  onSelect,
  onClear,
}: {
  lineage: Lineage;
  focusId: number;
  onSelect?: (id: number) => void;
  onClear?: () => void;
}) {
  const flowRef = useRef<ReactFlowInstance | null>(null);
  const closure = useMemo(() => ancestorClosure(lineage, focusId), [lineage, focusId]);
  const compressed = useMemo(() => compressEliteChains(closure), [closure]);
  const { nodes, edges } = useMemo(() => lineageToFlow(compressed), [compressed]);
  const [keyboardFocusId, setKeyboardFocusId] = useState(focusId);
  const [copyStatus, setCopyStatus] = useState("");
  const collapsedCount = closure.nodes.length - compressed.nodes.length;

  useEffect(() => {
    if (compressed.nodes.some((node) => node.id === focusId)) setKeyboardFocusId(focusId);
  }, [compressed.nodes, focusId]);

  const visibleNodes = useMemo(() => nodes.map((node) => ({
    ...node,
    selected: Number(node.id) === keyboardFocusId,
  })), [keyboardFocusId, nodes]);

  async function copyFocusedExpression() {
    const node = compressed.nodes.find((item) => item.id === keyboardFocusId);
    if (!node) return;
    try {
      await navigator.clipboard.writeText(serializeFormula(node.tree));
      setCopyStatus(`Copied formula ${node.id}.`);
    } catch {
      setCopyStatus("Clipboard access is unavailable.");
    }
  }

  return (
    <div
      className="graph ancestry-view"
      data-testid="ancestry-view"
      role="region"
      aria-label={`Ancestry graph for formula node ${focusId}`}
      aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight Enter Space Control+C Meta+C 0 Escape"
      tabIndex={0}
      onKeyDown={(event) => {
        if (
          event.key.toLowerCase() === "c" &&
          (event.ctrlKey || event.metaKey)
        ) {
          event.preventDefault();
          void copyFocusedExpression();
        } else if (event.key === "0") {
          event.preventDefault();
          void flowRef.current?.fitView({ padding: 0.18 });
        } else if (
          event.key === "ArrowUp" ||
          event.key === "ArrowDown" ||
          event.key === "ArrowLeft" ||
          event.key === "ArrowRight"
        ) {
          event.preventDefault();
          setKeyboardFocusId(lineageKeyboardNeighbor(
            compressed,
            keyboardFocusId,
            event.key.replace("Arrow", "").toLowerCase() as "up" | "down" | "left" | "right",
          ));
        } else if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect?.(keyboardFocusId);
        } else if (event.key === "Escape") {
          event.preventDefault();
          setKeyboardFocusId(focusId);
          onClear?.();
        }
      }}
    >
      <p className="ancestry-caption">
        Showing the {closure.nodes.length} ancestor(s) of node #{focusId}.
        {collapsedCount > 0 ? ` ${collapsedCount} unchanged elite retention steps are collapsed.` : ""}
      </p>
      <ReactFlow
        nodes={visibleNodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        fitView
        minZoom={0.25}
        maxZoom={1.8}
        zoomOnScroll
        zoomOnPinch
        panOnDrag
        nodesDraggable={false}
        nodesConnectable={false}
        deleteKeyCode={null}
        onInit={(instance) => {
          flowRef.current = instance as unknown as ReactFlowInstance;
        }}
        onNodeClick={(_, node) => {
          setKeyboardFocusId(Number(node.id));
          onSelect?.(Number(node.id));
        }}
        onPaneClick={() => onClear?.()}
      >
        <Background />
      </ReactFlow>
      <span className="visually-hidden" role="status">{copyStatus}</span>
    </div>
  );
}
