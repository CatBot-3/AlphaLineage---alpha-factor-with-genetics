import { useCallback, useEffect, useMemo, useState } from "react";
import type { FactorNode } from "../api/types";
import type { FormulaNodeData } from "../extend/formulaGraph";
import { serializeFormula } from "../extend/formulaText";
import { ResizableGraphWorkspace } from "../graph/ResizableGraphWorkspace";
import { ReadOnlyFormulaGraph } from "./ReadOnlyFormulaGraph";

type ResultMode = "visual" | "expression";

function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

function FormulaInspector({ node }: { node: FormulaNodeData | null }) {
  if (!node) {
    return (
      <div className="formula-result-inspector__empty">
        <h2>Inspector</h2>
        <p>Select a block to inspect its pinned, read-only definition.</p>
      </div>
    );
  }
  const numericEntries = Object.entries(node.inlineValues ?? {});
  return (
    <div data-testid="formula-result-inspector">
      <h2>Inspector</h2>
      <p className="formula-result-inspector__name">{node.label}</p>
      {node.description && <p className="hint">{node.description}</p>}
      <dl className="formula-result-inspector__facts">
        <div><dt>Block type</dt><dd>{node.kind}</dd></div>
        <div><dt>Output</dt><dd>{node.outType}</dd></div>
        {node.primitiveName && <div><dt>Runtime</dt><dd><code>{node.primitiveName}</code></dd></div>}
        {node.origin && <div><dt>Origin</dt><dd>{node.origin.replace(/_/g, " ")}</dd></div>}
        {node.revision != null && <div><dt>Pinned revision</dt><dd>v{node.revision}</dd></div>}
        {node.value != null && <div><dt>Value</dt><dd>{node.value}</dd></div>}
        {numericEntries.map(([index, value]) => (
          <div key={index}>
            <dt>{node.inputNames?.[Number(index)] ?? `Parameter ${Number(index) + 1}`}</dt>
            <dd>{value ?? "Unbound"}</dd>
          </div>
        ))}
      </dl>
      <p className="formula-result-inspector__readonly">This result is immutable. Open a copy to edit it.</p>
    </div>
  );
}

export function BestFormulaResultPage({
  factor,
  canSave,
  saved,
  onSave,
  onOpenCopy,
  onLegacySelection,
}: {
  factor: FactorNode;
  canSave: boolean;
  saved: boolean;
  onSave: () => void;
  onOpenCopy: () => void;
  onLegacySelection?: (node: { name: string; value?: number } | null) => void;
}) {
  const [mode, setMode] = useState<ResultMode>("visual");
  const [selectedNode, setSelectedNode] = useState<FormulaNodeData | null>(null);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  const expression = useMemo(() => serializeFormula(factor), [factor]);

  const inspect = useCallback((node: FormulaNodeData | null) => {
    setSelectedNode(node);
    onLegacySelection?.(node ? {
      name: node.primitiveName ?? node.label,
      value: typeof node.value === "number" ? node.value : undefined,
    } : null);
  }, [onLegacySelection]);

  const copyExpression = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(expression);
      setCopyStatus("Expression copied.");
    } catch {
      setCopyStatus("Clipboard access is unavailable. Select the expression to copy it.");
    }
  }, [expression]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (
        event.key.toLowerCase() !== "c" ||
        (!event.ctrlKey && !event.metaKey) ||
        isTextEntryTarget(event.target) ||
        window.getSelection()?.toString()
      ) return;
      event.preventDefault();
      void copyExpression();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [copyExpression]);

  return (
    <div className="best-formula-result" data-testid="best-formula-result">
      <div className="result-toolbar">
        <div>
          <h2>Selected formula snapshot</h2>
          <p>Inspect the exact formula selected for this round. Editing always starts from a copy.</p>
        </div>
        <div className="result-toolbar__actions">
          {canSave && (
            <button
              type="button"
              className="primary-action"
              data-testid="save-best-factor"
              disabled={saved}
              onClick={onSave}
            >
              {saved ? "Saved" : "Save"}
            </button>
          )}
          <button type="button" onClick={onOpenCopy}>Open copy</button>
          <button type="button" onClick={() => void copyExpression()}>Copy expression</button>
        </div>
        <div className="formula-mode result-mode" role="tablist" aria-label="Best Formula Result view">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "visual"}
            tabIndex={mode === "visual" ? 0 : -1}
            onClick={() => setMode("visual")}
          >
            Visual
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "expression"}
            tabIndex={mode === "expression" ? 0 : -1}
            onClick={() => setMode("expression")}
          >
            Expression
          </button>
        </div>
      </div>

      {copyStatus && <p className="surface-message" role="status">{copyStatus}</p>}

      {mode === "visual" ? (
        <ResizableGraphWorkspace
          storageKey="alphalineage:best-formula-layout"
          center={{
            id: "best-formula-canvas",
            label: "Best formula diagram",
            content: (
              <ReadOnlyFormulaGraph
                factor={factor}
                onInspect={inspect}
                ariaLabel="Best formula result diagram"
              />
            ),
          }}
          right={{
            id: "best-formula-inspector",
            label: "Formula inspector",
            content: <FormulaInspector node={selectedNode} />,
          }}
        />
      ) : (
        <section className="formula-result-expression" aria-label="Formula expression">
          <h2>Expression</h2>
          <pre tabIndex={0}><code>{expression}</code></pre>
          <p className="hint">This expression is read-only and identifies the selected snapshot.</p>
        </section>
      )}
    </div>
  );
}
