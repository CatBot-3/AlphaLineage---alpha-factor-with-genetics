// P6-T1: render a factor's expression tree with React Flow; clicking a node selects it.

import { Background, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import type { FactorNode } from "../api/types";
import { NodeCard } from "./NodeCard";
import { treeToFlow, type TreeNodeData } from "./treeToFlow";

const nodeTypes = { factorNode: NodeCard };

export function FactorTree({
  factor,
  onSelect,
}: {
  factor: FactorNode;
  onSelect?: (data: TreeNodeData) => void;
}) {
  const { nodes, edges } = useMemo(() => treeToFlow(factor), [factor]);
  return (
    <div className="graph" data-testid="factor-tree">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        zoomOnScroll
        zoomOnPinch
        panOnDrag
        onNodeClick={(_, node) => onSelect?.(node.data as TreeNodeData)}
      >
        <Background />
      </ReactFlow>
    </div>
  );
}
