// Custom React Flow node: renders a primitive's name (+ ephemeral value) as testable DOM.

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { formatValue, type TreeNodeData } from "./treeToFlow";

export function NodeCard({ data }: NodeProps) {
  const node = data as TreeNodeData;
  return (
    <div className="node-card" data-testid="factor-node">
      <Handle type="target" position={Position.Top} />
      <span className="node-name">{node.name}</span>
      {node.value !== undefined && (
        <span className="node-value"> = {formatValue(node.value)}</span>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
