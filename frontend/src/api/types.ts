// TypeScript mirrors of the backend JSON shapes (core/tree.py, library/store.py, pipeline.py).

export type AppMode = "demo" | "app";

export interface FactorNode {
  name: string;
  value?: number;
  children?: FactorNode[];
}

export interface Report {
  oos_ic: number;
  deflated_sharpe: number;
  pbo: number;
  train_ic: number;
  n_trials: number;
  significant: boolean;
}

export interface LineageNode {
  id: number;
  generation: number;
  op: string;
  parents: number[];
  tree: FactorNode;
}

export interface Lineage {
  run_id: string;
  metadata: Record<string, unknown>;
  nodes: LineageNode[];
}

export interface HistoryPoint {
  generation: number;
  best_fitness: number;
  mean_fitness: number;
  best_ic: number;
}

export interface RunResult {
  best_factor: string | FactorNode; // the service returns a JSON string; demo JSON may store a tree
  report: Report;
  generations: number;
  history: HistoryPoint[];
  lineage: Lineage;
}

export function parseFactor(factor: string | FactorNode): FactorNode {
  return typeof factor === "string" ? (JSON.parse(factor) as FactorNode) : factor;
}

// --- extensibility (Phase 7) ---------------------------------------------------
export interface PrimitiveInfo {
  name: string;
  kind: string; // operator | operand | ephemeral
  arg_types: string[];
  out_type: string;
  user: boolean;
}

export interface OperatorSpec {
  name: string;
  arg_types: string[];
  out_type: string;
  body: FactorNode; // a typed body tree with $arg leaves
}

export interface UniverseMembership {
  symbol: string;
  entry: string;
  exit?: string | null;
}

export interface UniverseSpec {
  name: string;
  memberships: UniverseMembership[];
}

export interface UniverseInfo extends UniverseSpec {
  symbols: string[];
  source: "sample" | "custom";
}

export interface UniverseDraft {
  name: string;
  rows: Array<{ symbol: string; entry: string; exit: string }>;
}

export interface OperatorDraftNode {
  id: string;
  kind: string;
  label?: string;
  argIndex?: number;
  value?: number;
  x: number;
  y: number;
}

export interface OperatorDraftEdge {
  source: string;
  target: string;
}

export interface OperatorComposerDraft {
  name: string;
  argTypes: string[];
  outType: string;
  nodes: OperatorDraftNode[];
  edges: OperatorDraftEdge[];
}

export interface WorkspaceUiState {
  selectedTab?: "dashboard" | "factor" | "genealogy" | "extend";
  selectedFactorNode?: { name: string; value?: number } | null;
  selectedLineage?: number | null;
}

export interface WorkspaceSnapshot {
  id?: string;
  name: string;
  version: 1;
  savedAt: string;
  run: RunResult | null;
  universes: UniverseSpec[];
  operators: OperatorSpec[];
  universeDraft?: UniverseDraft;
  operatorDraft?: OperatorComposerDraft;
  ui: WorkspaceUiState;
}

export interface WorkspaceSummary {
  id: string;
  name: string;
  savedAt: string;
  hasRun: boolean;
}
