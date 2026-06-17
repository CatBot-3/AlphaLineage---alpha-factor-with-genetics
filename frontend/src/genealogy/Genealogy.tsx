// B4: the genealogy view. Default is the grouped, collapsible generation list (Design A);
// "Trace ancestry" switches to a focused ancestor-closure DAG (Design B). The old single
// whole-run DAG is gone - it was illegible past a few hundred nodes.

import { useMemo, useState } from "react";
import type { Lineage, LineageNode } from "../api/types";
import { AncestryView } from "./AncestryView";
import { bestFinalNode } from "./ancestry";
import { GenerationList } from "./GenerationList";
import { groupLineage } from "./groupLineage";

type Mode = "list" | "ancestry";

export function Genealogy({
  lineage,
  onSelect,
  onSave,
}: {
  lineage: Lineage;
  onSelect?: (id: number) => void;
  onSave?: (node: LineageNode) => void;
}) {
  const [mode, setMode] = useState<Mode>("list");
  const [focusId, setFocusId] = useState<number | null>(null);
  const generations = useMemo(() => groupLineage(lineage), [lineage]);

  if (!lineage?.nodes?.length) {
    return (
      <div className="genealogy" data-testid="genealogy">
        <p className="hint">No lineage to display for this run.</p>
      </div>
    );
  }

  function trace(id: number) {
    setFocusId(id);
    setMode("ancestry");
    onSelect?.(id);
  }

  const effectiveFocus = focusId ?? bestFinalNode(lineage);

  return (
    <div className="genealogy" data-testid="genealogy">
      <div className="genealogy-modes" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "list"}
          onClick={() => setMode("list")}
        >
          Generations
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "ancestry"}
          data-testid="mode-ancestry"
          onClick={() => {
            if (focusId === null) setFocusId(bestFinalNode(lineage));
            setMode("ancestry");
          }}
        >
          Trace ancestry
        </button>
      </div>

      {mode === "list" ? (
        <GenerationList
          generations={generations}
          lineage={lineage}
          onSelect={(id) => onSelect?.(id)}
          onTrace={trace}
          onSave={onSave}
        />
      ) : effectiveFocus !== null ? (
        <AncestryView
          lineage={lineage}
          focusId={effectiveFocus}
          onSelect={(id) => onSelect?.(id)}
        />
      ) : (
        <p className="hint">No lineage to trace yet.</p>
      )}
    </div>
  );
}
