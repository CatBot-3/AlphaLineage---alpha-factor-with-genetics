// Navigable lineage detail: trace a node's parents -> operation -> children. Clicking a
// parent or child selects it, so the user can walk the genealogy.

import type { Lineage } from "../api/types";

export function LineageDetail({
  lineage,
  selectedId,
  onSelect,
}: {
  lineage: Lineage;
  selectedId: number | null;
  onSelect: (id: number) => void;
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
        node #{node.id} · gen {node.generation}
      </h4>
      <p>
        operation: <b data-testid="lineage-op">{node.op}</b>
      </p>
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
