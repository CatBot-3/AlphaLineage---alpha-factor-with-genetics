// B4 Design B (trace ancestry): render only a chosen node's ancestor closure as a React Flow
// DAG. The closure is tens of nodes at most, so a graph genuinely beats a list here - it answers
// "where did this factor come from?" spatially.

import { Background, Controls, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import type { Lineage } from "../api/types";
import { ancestorClosure } from "./ancestry";
import { lineageToFlow } from "./lineageToFlow";

export function AncestryView({
  lineage,
  focusId,
  onSelect,
}: {
  lineage: Lineage;
  focusId: number;
  onSelect?: (id: number) => void;
}) {
  const closure = useMemo(() => ancestorClosure(lineage, focusId), [lineage, focusId]);
  const { nodes, edges } = useMemo(() => lineageToFlow(closure), [closure]);

  return (
    <div className="graph ancestry-view" data-testid="ancestry-view">
      <p className="ancestry-caption">
        Showing the {closure.nodes.length} ancestor(s) of node #{focusId}.
      </p>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        onNodeClick={(_, node) => onSelect?.(Number(node.id))}
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}
