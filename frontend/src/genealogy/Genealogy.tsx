import { useMemo, useState } from "react";
import type { Lineage, LineageNode } from "../api/types";
import { serializeFormula } from "../extend/formulaText";
import { ReadOnlyFormulaGraph } from "../factor/ReadOnlyFormulaGraph";
import { ResizableGraphWorkspace } from "../graph/ResizableGraphWorkspace";
import { AncestryView } from "./AncestryView";
import { bestFinalNode } from "./ancestry";
import { GenerationList } from "./GenerationList";
import { groupLineage } from "./groupLineage";
import { LineageDetail } from "./LineageDetail";

export function Genealogy({
  lineage,
  selectedId,
  onSelect,
  onSave,
}: {
  lineage: Lineage;
  selectedId?: number | null;
  onSelect?: (id: number) => void;
  onSave?: (node: LineageNode) => void;
}) {
  const selectedByRound = Number(lineage.metadata?.selected_lineage_node_id);
  const defaultFocus =
    Number.isInteger(selectedByRound) && lineage.nodes.some((node) => node.id === selectedByRound)
      ? selectedByRound
      : bestFinalNode(lineage);
  const [localSelection, setLocalSelection] = useState<number | null>(defaultFocus);
  const requestedSelection = selectedId === undefined ? localSelection : selectedId;
  const effectiveFocus =
    requestedSelection !== null && lineage.nodes.some((node) => node.id === requestedSelection)
      ? requestedSelection
      : defaultFocus;
  const generations = useMemo(() => groupLineage(lineage), [lineage]);
  const selectedNode = effectiveFocus === null
    ? null
    : lineage.nodes.find((node) => node.id === effectiveFocus) ?? null;

  if (!lineage?.nodes?.length) {
    return (
      <div className="genealogy" data-testid="genealogy">
        <p className="hint">No lineage to display for this round.</p>
      </div>
    );
  }

  function select(id: number) {
    setLocalSelection(id);
    onSelect?.(id);
  }

  return (
    <div className="genealogy" data-testid="genealogy">
      <ResizableGraphWorkspace
        storageKey="alphalineage:genealogy-layout"
        leftDefault={300}
        rightDefault={360}
        left={{
          id: "genealogy-generations",
          label: "Generations",
          content: (
            <section className="genealogy-pane">
              <header className="genealogy-pane__header">
                <h2>Generations</h2>
                <p>Select a formula to trace the ancestry that produced it.</p>
              </header>
              <GenerationList
                generations={generations}
                lineage={lineage}
                onSelect={select}
                onTrace={select}
                onSave={onSave}
              />
            </section>
          ),
        }}
        center={{
          id: "genealogy-ancestry",
          label: "Selected formula ancestry",
          content: effectiveFocus !== null ? (
            <section className="genealogy-canvas">
              <header className="genealogy-canvas__header">
                <div>
                  <h2>Ancestry of formula #{effectiveFocus}</h2>
                  <p>
                    Wheel or pinch to zoom. Arrow keys move focus; Enter inspects; Ctrl/Cmd+C
                    copies; 0 fits the graph.
                  </p>
                </div>
              </header>
              <AncestryView
                lineage={lineage}
                focusId={effectiveFocus}
                onSelect={select}
              />
            </section>
          ) : (
            <p className="hint">No lineage to trace yet.</p>
          ),
        }}
        right={{
          id: "genealogy-inspector",
          label: "Formula lineage inspector",
          content: (
            <section className="genealogy-inspector">
              <header className="genealogy-pane__header">
                <h2>Inspector</h2>
                <p>Lineage metadata and the immutable formula snapshot.</p>
              </header>
              <LineageDetail
                lineage={lineage}
                selectedId={effectiveFocus}
                onSelect={select}
                onSave={onSave}
              />
              {selectedNode && (
                <section className="genealogy-formula-preview" data-testid="genealogy-formula-preview">
                  <h3>Selected formula</h3>
                  <ReadOnlyFormulaGraph
                    factor={selectedNode.tree}
                    compact
                    ariaLabel={`Formula preview for lineage node ${selectedNode.id}`}
                  />
                  <code className="genealogy-formula-preview__expression">
                    {serializeFormula(selectedNode.tree)}
                  </code>
                </section>
              )}
            </section>
          ),
        }}
      />
    </div>
  );
}
