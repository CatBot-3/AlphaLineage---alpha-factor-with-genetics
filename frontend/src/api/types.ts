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
  fitness?: number | null; // present for runs >= the session/genealogy redesign; null for old demos
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
  // session-aware fields (present for runs launched as a session segment)
  session_id?: string;
  segment?: number;
  test_reads?: number;
  cumulative_trials?: number;
  repeated_oos_warning?: boolean;
}

// --- iterative sessions (A4/A5) ------------------------------------------------
export interface GpConfig {
  population_size: number;
  generations: number;
  tournament_size: number;
  crossover_rate: number;
  subtree_mutation_rate: number;
  point_mutation_rate: number;
  max_depth: number;
  max_nodes: number;
  parsimony: number;
  elitism: number;
  ic_method: string;
  min_names: number;
  horizon: number;
  min_depth: number;
  seed: number;
  time_budget_s: number | null;
}

export interface ProgressSnapshot {
  phase: string;
  generation: number;
  target_generations: number;
  history: Array<{ generation: number; best_fitness: number; mean_fitness: number }>;
  best: { tree: string; fitness: number } | null;
}

export interface SessionBoundaries {
  train_end: string;
  valid_start: string;
  valid_end: string;
  test_start: string;
  embargo: number;
}

export interface SessionSegment {
  index: number;
  universe: string;
  config: Partial<GpConfig>;
  gen_start: number;
  gen_end: number;
  new_trials: number;
  status: string;
}

export interface SessionJob {
  id: string;
  status: string; // queued | running | done | failed
  progress: ProgressSnapshot | null;
}

export interface SessionState {
  id: string;
  name: string;
  created_at: string;
  universe: string;
  as_of: string;
  boundaries: SessionBoundaries;
  config: Partial<GpConfig>;
  operators: OperatorSpec[];
  seed_factor_ids: string[];
  trial_baseline: number;
  cumulative_trials: number;
  test_reads: number;
  segments: SessionSegment[];
  last_job_id: string | null;
  job: SessionJob | null;
  result: RunResult | null;
}

export interface SessionSummary {
  id: string;
  name: string;
  created_at: string;
  universe: string;
  segments: number;
  cumulative_trials: number;
  test_reads: number;
}

export interface SessionCreateRequest {
  name?: string;
  universe?: string;
  as_of?: string;
  config?: Partial<GpConfig>;
  operators?: OperatorSpec[];
  seed_factor_ids?: string[];
  train?: number;
  valid?: number;
  embargo?: number;
}

export interface SessionContinueRequest {
  generations: number;
  config?: Partial<GpConfig>;
  universe?: string | null;
  operators?: OperatorSpec[];
  seed_factor_ids?: string[];
}

// --- saved factors (A3) --------------------------------------------------------
export interface FactorProvenance {
  session_id?: string;
  generation?: number;
  universe?: string;
  cumulative_trials?: number;
  test_reads?: number;
  test_start?: string;
}

export interface SavedFactor {
  id: string;
  name: string;
  saved_at: string;
  tree: FactorNode;
  metrics: Record<string, number>;
  provenance: FactorProvenance;
  required_operators: OperatorSpec[];
  notes: string;
  disclaimer: string;
}

export interface Settings {
  factors_dir: string;
  tiingo_api_key_set: boolean;
  evaluator: "auto" | "python" | "cpp";
  cpp_available: boolean;
}

export interface SettingsUpdate {
  factors_dir?: string;
  tiingo_api_key?: string;
  evaluator?: "auto" | "python" | "cpp";
}

export interface DataUsageRow {
  key: string;
  label: string;
  bytes: number;
  count: number;
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
  selectedTab?: "train" | "dashboard" | "factor" | "genealogy" | "extend" | "library";
  selectedFactorNode?: { name: string; value?: number } | null;
  selectedLineage?: number | null;
  sessionId?: string | null;
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
