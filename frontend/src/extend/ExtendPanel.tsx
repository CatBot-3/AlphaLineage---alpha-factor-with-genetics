// Extend tab: a dropdown switching between the three Extend pages, replacing the old
// side-by-side universe/formula layout.

import { useState } from "react";
import type {
  FormulaDraft,
  OperatorComposerDraft,
  SyncProgressSnapshot,
  UniverseDraft,
} from "../api/types";
import { FormulaEditorPage } from "./FormulaEditorPage";
import { SyncDataPage } from "./SyncDataPage";
import { UniverseEditorPage } from "./UniverseEditorPage";

export type ExtendPage = "formula" | "universe" | "sync";

const PAGE_LABELS: Record<ExtendPage, string> = {
  universe: "Universe Editor",
  sync: "Sync Data",
  formula: "Formula Editor",
};

export function ExtendPanel({
  initialPage = "universe",
  universeDraft,
  onUniverseDraftChange,
  formulaDraft,
  onFormulaDraftChange,
  operatorDraft,
  onOperatorDraftChange,
  canSubmit = true,
  onDataPullProgressChange,
}: {
  initialPage?: ExtendPage;
  universeDraft?: UniverseDraft;
  onUniverseDraftChange?: (draft: UniverseDraft) => void;
  formulaDraft?: FormulaDraft;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
  operatorDraft?: OperatorComposerDraft;
  onOperatorDraftChange?: (draft: OperatorComposerDraft) => void;
  canSubmit?: boolean;
  onDataPullProgressChange?: (snapshot: SyncProgressSnapshot | null) => void;
}) {
  const [page, setPage] = useState<ExtendPage>(initialPage);
  const rows = universeDraft?.rows ?? [];

  return (
    <div className="extend-panel">
      <label className="field extend-page-picker">
        <span className="field-label">Extend</span>
        <select
          aria-label="Extend page"
          value={page}
          onChange={(event) => setPage(event.target.value as ExtendPage)}
        >
          {(Object.keys(PAGE_LABELS) as ExtendPage[]).map((id) => (
            <option key={id} value={id}>
              {PAGE_LABELS[id]}
            </option>
          ))}
        </select>
      </label>

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
          operatorDraft={operatorDraft}
          onOperatorDraftChange={onOperatorDraftChange}
          canSubmit={canSubmit}
        />
      )}
    </div>
  );
}
