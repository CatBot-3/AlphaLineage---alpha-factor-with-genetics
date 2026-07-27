import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { LineageNodeData } from "./lineageToFlow";

export function LineageNodeCard({ data, selected }: NodeProps) {
  const node = data as LineageNodeData;
  return (
    <div
      className={`lineage-node-card${selected ? " is-selected" : ""}`}
      data-testid="lineage-node"
      aria-label={`${node.label}, generation ${node.generation}, ${node.op}`}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <strong>{node.label}</strong>
      <span>Generation {node.generation}</span>
      <small>{node.op}</small>
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}
