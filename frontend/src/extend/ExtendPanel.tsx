// Extend tab body: renders whichever of the three Extend pages the nav dropdown selected.
// The active page is controlled by App (via the `page` prop); the dropdown lives in the nav.

import type { FormulaDraft, SyncProgressSnapshot, UniverseDraft } from "../api/types";
import { FormulaEditorPage } from "./FormulaEditorPage";
import { SyncDataPage } from "./SyncDataPage";
import { UniverseEditorPage } from "./UniverseEditorPage";

export type ExtendPage = "formula" | "universe" | "sync";
type FormulaDraftRecovery = NonNullable<FormulaDraft["recoveredDrafts"]>[number];

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
  page: ExtendPage;
  universeDraft?: UniverseDraft;
  onUniverseDraftChange?: (draft: UniverseDraft) => void;
  formulaDraft?: FormulaDraft;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
  recoveredFormulaDrafts?: FormulaDraftRecovery[];
  onRecoverFormulaDraft?: (index: number) => void;
  onOpenDataSync?: () => void;
  canSubmit?: boolean;
  onDataPullProgressChange?: (snapshot: SyncProgressSnapshot | null) => void;
}) {
  const rows = universeDraft?.rows ?? [];

  return (
    <div className="extend-panel">
      {page === "universe" && (
        <UniverseEditorPage
          draft={universeDraft}
          onDraftChange={onUniverseDraftChange}
          canSubmit={canSubmit}
          onPullProgress={onDataPullProgressChange}
        />
      )}
      {page === "sync" && (
        <SyncDataPage rows={rows} canSubmit={canSubmit} onPullProgress={onDataPullProgressChange} />
      )}
      {page === "formula" && (
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
