// Factor detail panel: shows the selected node's primitive and its parameter.

import { formatValue, type TreeNodeData } from "./treeToFlow";

export function FactorDetail({ node }: { node: TreeNodeData | null }) {
  if (!node) {
    return <div className="detail">Click a node to inspect its parameters.</div>;
  }
  return (
    <div className="detail" data-testid="factor-detail">
      <h4>{node.name}</h4>
      {node.value !== undefined ? (
        <p>
          parameter: <b>{formatValue(node.value)}</b>
        </p>
      ) : (
        <p>operator / operand (no tunable parameter)</p>
      )}
    </div>
  );
}
