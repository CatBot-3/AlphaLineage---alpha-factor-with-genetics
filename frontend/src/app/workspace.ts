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
  recoveredFormulaDrafts?: FormulaDraftRecovery[];
  operatorDraft?: OperatorComposerDraft;
}

export type FormulaDraftRecovery = NonNullable<FormulaDraft["recoveredDrafts"]>[number];

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
    formulaDraft: bundleFormulaDrafts(input.formulaDraft, input.recoveredFormulaDrafts),
    operatorDraft: input.operatorDraft,
    ui: input.ui,
  };
}

function withoutLegacyBuilderState(draft: FormulaDraft | undefined): FormulaDraft | undefined {
  if (!draft) return undefined;
  const legacy = draft as FormulaDraft & { builderMode?: "factor" | "reusable" };
  const {
    builderDrafts: _builderDrafts,
    recoveredDrafts: _recoveredDrafts,
    builderMode: _builderMode,
    ...plain
  } = legacy;
  return plain;
}

function isLegacyStarterBody(body: FormulaDraft["body"]): boolean {
  if (!body) return true;
  return body.name === "rank" && body.children?.length === 1 &&
    body.children[0]?.name === "$arg" && Number(body.children[0]?.value) === 0;
}

function isLegacyStarterGraph(draft: FormulaDraft): boolean {
  const nodes = draft.graphNodes ?? [];
  const edges = draft.graphEdges ?? [];
  if (nodes.length === 0) return true;
  const input = nodes.filter((node) => node.data.kind === "input");
  const output = nodes.filter((node) => node.data.kind === "output");
  const calculations = nodes.filter((node) => (
    node.data.kind !== "input" && node.data.kind !== "output"
  ));
  if (input.length !== 1 || output.length !== 1) return false;
  if (calculations.length === 0) return edges.length === 0;
  if (calculations.length !== 1) return false;
  const calculation = calculations[0];
  const primitive = calculation.data.primitiveName ?? calculation.data.logicalName;
  return primitive === "rank" && edges.length === 2 &&
    edges.some((edge) => edge.source === input[0].id && edge.target === calculation.id) &&
    edges.some((edge) => edge.source === calculation.id && edge.target === output[0].id);
}

/** Remove only the untouched starter that older builders inserted automatically. */
function withoutUntouchedLegacyStarter(draft: FormulaDraft | undefined): FormulaDraft | undefined {
  const clean = withoutLegacyBuilderState(draft);
  if (!clean) return undefined;
  const inputs = clean.inputs ?? [];
  const starterInput = inputs.length === 1 && inputs[0].name === "price" &&
    inputs[0].type === "series" &&
    (!inputs[0].description || inputs[0].description === "Price or derived series to transform.");
  const untouchedIdentity = clean.name === "my_formula" && clean.display_name === "My formula" &&
    clean.description === "" && !clean.loadedName && !clean.loadedRevision &&
    (!clean.expression || clean.expression === "rank($price)");
  if (!starterInput || !untouchedIdentity || !isLegacyStarterBody(clean.body) ||
      !isLegacyStarterGraph(clean)) return clean;
  return {
    name: "my_formula",
    display_name: "My formula",
    description: "",
    inputs: [],
    arg_types: [],
    out_type: "signal",
    category: clean.category ?? "custom",
    activeMode: clean.activeMode ?? "visual",
  };
}

export interface MigratedFormulaDrafts {
  active?: FormulaDraft;
  recoveries: FormulaDraftRecovery[];
}

/**
 * Collapse the former Factor Builder / Reusable Formula Builder drafts into one editor draft.
 * When both existed, the reusable draft remains active and the factor draft is retained as a
 * recoverable payload. A workspace containing only either old draft opens it directly.
 */
export function migrateFormulaDrafts(stored: FormulaDraft | undefined): MigratedFormulaDrafts {
  if (!stored) return { recoveries: [] };
  const recoveries = (stored.recoveredDrafts ?? []).map((entry) => ({
    label: entry.label,
    draft: withoutUntouchedLegacyStarter(entry.draft)!,
  }));
  const reusable = stored.builderDrafts?.reusable;
  const factor = stored.builderDrafts?.factor;
  if (reusable) {
    return {
      active: withoutUntouchedLegacyStarter(reusable),
      recoveries: [
        ...(factor
          ? [{ label: "Legacy Factor Builder draft", draft: withoutUntouchedLegacyStarter(factor)! }]
          : []),
        ...recoveries,
      ],
    };
  }
  if (factor) return { active: withoutUntouchedLegacyStarter(factor), recoveries };
  return { active: withoutUntouchedLegacyStarter(stored), recoveries };
}

export function bundleFormulaDrafts(
  active: FormulaDraft | undefined,
  recoveries: FormulaDraftRecovery[] | undefined,
): FormulaDraft | undefined {
  const cleanActive = withoutLegacyBuilderState(active);
  const cleanRecoveries = (recoveries ?? []).map((entry) => ({
    label: entry.label,
    draft: withoutLegacyBuilderState(entry.draft)!,
  }));
  if (!cleanActive && cleanRecoveries.length === 0) return undefined;
  if (!cleanActive) {
    const [promoted, ...remaining] = cleanRecoveries;
    return remaining.length > 0
      ? { ...promoted.draft, recoveredDrafts: remaining }
      : promoted.draft;
  }
  if (cleanRecoveries.length === 0) return cleanActive;
  return {
    ...cleanActive,
    recoveredDrafts: cleanRecoveries,
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
