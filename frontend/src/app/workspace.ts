import type {
  FormulaDraft,
  OperatorComposerDraft,
  RunResult,
  UniverseDraft,
  UniverseSpec,
  WorkspaceSnapshot,
  WorkspaceUiState,
} from "../api/types";
import { toUniversePayload } from "../extend/toUniversePayload";

export const WORKSPACE_KEY = "alphalineage:workspace:v1";

export interface WorkspaceInput {
  run: RunResult | null;
  ui: WorkspaceUiState;
  universeDraft?: UniverseDraft;
  formulaDraft?: FormulaDraft;
  operatorDraft?: OperatorComposerDraft;
}

function hasStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function universesFromDraft(draft?: UniverseDraft): UniverseSpec[] {
  if (!draft) return [];
  const spec = toUniversePayload(draft.name, draft.rows);
  return spec.name && spec.memberships.length > 0 ? [spec] : [];
}

export function makeWorkspaceSnapshot(input: WorkspaceInput): WorkspaceSnapshot {
  return {
    id: "local-workspace",
    name: "Local Workspace",
    version: 1,
    savedAt: new Date().toISOString(),
    run: input.run,
    universes: universesFromDraft(input.universeDraft),
    operators: [],
    universeDraft: input.universeDraft,
    formulaDraft: input.formulaDraft,
    operatorDraft: input.operatorDraft,
    ui: input.ui,
  };
}

export function readLocalWorkspace(): WorkspaceSnapshot | null {
  if (!hasStorage()) return null;
  try {
    const raw = window.localStorage.getItem(WORKSPACE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<WorkspaceSnapshot>;
    if (parsed.version !== 1 || typeof parsed.name !== "string") return null;
    return parsed as WorkspaceSnapshot;
  } catch {
    return null;
  }
}

export function writeLocalWorkspace(snapshot: WorkspaceSnapshot): void {
  if (!hasStorage()) return;
  try {
    window.localStorage.setItem(WORKSPACE_KEY, JSON.stringify(snapshot));
  } catch {
    // Storage can be disabled or quota-limited; the app should keep running.
  }
}

export function clearLocalWorkspace(): void {
  if (!hasStorage()) return;
  try {
    window.localStorage.removeItem(WORKSPACE_KEY);
  } catch {
    // noop
  }
}
