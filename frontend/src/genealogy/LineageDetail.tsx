// Navigable lineage detail: trace a node's parents -> operation -> children.

import type { Lineage, LineageNode } from "../api/types";

export function LineageDetail({
  lineage,
  selectedId,
  onSelect,
  onSave,
}: {
  lineage: Lineage;
  selectedId: number | null;
  onSelect: (id: number) => void;
  onSave?: (node: LineageNode) => void;
}) {
  if (selectedId === null) {
    return <div className="detail">Select a node to trace its lineage.</div>;
  }
  const node = lineage.nodes.find((n) => n.id === selectedId);
  if (!node) {
    return <div className="detail">Unknown node #{selectedId}.</div>;
  }
  const children = lineage.nodes.filter((n) => n.parents.includes(selectedId));

  return (
    <div className="detail" data-testid="lineage-detail">
      <h4>
        node #{node.id} / gen {node.generation}
      </h4>
      <p>
        operation: <b data-testid="lineage-op">{node.op}</b>
      </p>
      {typeof node.fitness === "number" && (
        <p>
          fitness: <b>{node.fitness.toFixed(4)}</b>
        </p>
      )}
      {onSave && (
        <button
          type="button"
          className="ghost"
          data-testid="save-lineage-node"
          onClick={() => onSave(node)}
        >
          Save to library
        </button>
      )}
      <div className="lineage-links">
        <span>parents:</span>{" "}
        {node.parents.length === 0 ? (
          <span>none (seed)</span>
        ) : (
          node.parents.map((p) => (
            <button key={p} data-testid="lineage-parent" onClick={() => onSelect(p)}>
              #{p}
            </button>
          ))
        )}
      </div>
      <div className="lineage-links">
        <span>children:</span>{" "}
        {children.length === 0 ? (
          <span>none</span>
        ) : (
          children.map((c) => (
            <button key={c.id} data-testid="lineage-child" onClick={() => onSelect(c.id)}>
              #{c.id}
            </button>
          ))
        )}
      </div>
    </div>
  );
}
