// P6-T2: render the evolution genealogy as a React Flow DAG; clicking a node selects it.

import { Background, Controls, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import type { Lineage } from "../api/types";
import { lineageToFlow } from "./lineageToFlow";

export function Genealogy({
  lineage,
  onSelect,
}: {
  lineage: Lineage;
  onSelect?: (id: number) => void;
}) {
  const { nodes, edges } = useMemo(() => lineageToFlow(lineage), [lineage]);
  return (
    <div className="graph" data-testid="genealogy">
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
