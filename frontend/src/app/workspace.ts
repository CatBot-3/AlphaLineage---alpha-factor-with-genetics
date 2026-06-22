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
import { findRoot, graphToBody } from "../extend/graphToBody";

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
    const snapshot = parsed as WorkspaceSnapshot;
    if (!snapshot.formulaDraft && snapshot.operatorDraft) {
      const old = snapshot.operatorDraft;
      const nodes = old.nodes.map((node) => ({
        id: node.id,
        kind: node.kind,
        argIndex: node.argIndex,
        value: node.value,
        x: node.x,
        y: node.y,
      }));
      const root = findRoot(nodes, old.edges);
      if (root) {
        const inputs = old.argTypes.map((type, index) => ({
          name: `input_${index + 1}`,
          type,
          description: "Migrated formula input.",
        }));
        snapshot.formulaDraft = {
          name: old.name,
          display_name: old.name.replace(/_/g, " "),
          description: "Migrated from the previous graph composer.",
          arg_types: old.argTypes,
          inputs,
          out_type: old.outType,
          body: graphToBody(nodes, old.edges, root),
          category: "custom",
          activeMode: "visual",
        };
      }
    }
    return snapshot;
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
