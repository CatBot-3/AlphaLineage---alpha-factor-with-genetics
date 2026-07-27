// Extend tab body: renders the unified Formula Builder or Universe Editor.
// The active page is controlled by App (via the `page` prop); the dropdown lives in the nav.

import type { FormulaDraft, SyncProgressSnapshot, UniverseDraft } from "../api/types";
import { FormulaEditorPage } from "./FormulaEditorPage";
import { UniverseEditorPage } from "./UniverseEditorPage";

export type ExtendPage = "formula" | "universe";
export type ExtendPageInput = ExtendPage | "sync";
type FormulaDraftRecovery = NonNullable<FormulaDraft["recoveredDrafts"]>[number];

/** Maps workspaces/navigation state saved before price sync moved into Universe Editor. */
export function normalizeExtendPage(page: ExtendPageInput): ExtendPage {
  return page === "sync" ? "universe" : page;
}

export function ExtendPanel({
  page,
  universeDraft,
  onUniverseDraftChange,
  formulaDraft,
  onFormulaDraftChange,
  recoveredFormulaDrafts = [],
  onRecoverFormulaDraft,
  onOpenDataSync,
  canSubmit = true,
  onDataPullProgressChange,
}: {
  page: ExtendPageInput;
  universeDraft?: UniverseDraft;
  onUniverseDraftChange?: (draft: UniverseDraft) => void;
  formulaDraft?: FormulaDraft;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
  recoveredFormulaDrafts?: FormulaDraftRecovery[];
  onRecoverFormulaDraft?: (index: number) => void;
  onOpenDataSync?: (universeName?: string) => void;
  canSubmit?: boolean;
  onDataPullProgressChange?: (snapshot: SyncProgressSnapshot | null) => void;
}) {
  const activePage = normalizeExtendPage(page);

  return (
    <div className="extend-panel">
      {activePage === "universe" && (
        <UniverseEditorPage
          draft={universeDraft}
          onDraftChange={onUniverseDraftChange}
          canSubmit={canSubmit}
          onPullProgress={onDataPullProgressChange}
        />
      )}
      {activePage === "formula" && (
        <>
          {recoveredFormulaDrafts.length > 0 && (
            <details className="surface-message" data-testid="recovered-formula-drafts">
              <summary>Recovered Formula Builder drafts ({recoveredFormulaDrafts.length})</summary>
              <p className="hint">
                These drafts came from the former separate builders. Opening one keeps the current
                draft available here, so no work is discarded.
              </p>
              {recoveredFormulaDrafts.map((entry, index) => (
                <button
                  key={`${entry.label}-${index}`}
                  type="button"
                  className="ghost"
                  onClick={() => onRecoverFormulaDraft?.(index)}
                >
                  Open {entry.label}
                </button>
              ))}
            </details>
          )}
          <FormulaEditorPage
            formulaDraft={formulaDraft}
            onFormulaDraftChange={onFormulaDraftChange}
            defaultUniverse={universeDraft?.selectedUniverse}
            onDataSync={onOpenDataSync}
            canSubmit={canSubmit}
          />
        </>
      )}
    </div>
  );
}
