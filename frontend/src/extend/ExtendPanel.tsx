// Extend tab body: renders whichever of the three Extend pages the nav dropdown selected.
// The active page is controlled by App (via the `page` prop); the dropdown lives in the nav.

import type { FormulaDraft, SyncProgressSnapshot, UniverseDraft } from "../api/types";
import { FormulaEditorPage } from "./FormulaEditorPage";
import { SyncDataPage } from "./SyncDataPage";
import { UniverseEditorPage } from "./UniverseEditorPage";

export type ExtendPage = "formula" | "universe" | "sync";

export function ExtendPanel({
  page,
  universeDraft,
  onUniverseDraftChange,
  formulaDraft,
  onFormulaDraftChange,
  canSubmit = true,
  onDataPullProgressChange,
}: {
  page: ExtendPage;
  universeDraft?: UniverseDraft;
  onUniverseDraftChange?: (draft: UniverseDraft) => void;
  formulaDraft?: FormulaDraft;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
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
        <FormulaEditorPage
          formulaDraft={formulaDraft}
          onFormulaDraftChange={onFormulaDraftChange}
          canSubmit={canSubmit}
        />
      )}
    </div>
  );
}
