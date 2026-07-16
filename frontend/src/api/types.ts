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
  oos_backtest?: RunBacktestReport;
  formula_revisions?: Array<{ runtime_name?: string; name?: string; revision?: number }>;
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
  formula_revisions?: Array<{ runtime_name?: string; name?: string; revision?: number }>;
  resources?: ResolvedTrainingResources | null;
  termination_reason?: "completed" | "user_stopped" | "time_budget";
  timings?: {
    training_seconds: number;
    reporting_seconds: number;
    total_seconds: number;
  };
  context?: RunContext;
  oos_backtest?: RunBacktestReport;
}

export interface RunContext {
  universe?: string;
  universe_revision?: string;
  as_of?: string;
  boundaries?: {
    train_end?: string;
    valid_start?: string;
    valid_end?: string;
    test_start?: string;
    test_end?: string;
    embargo?: number;
  };
  horizon?: number;
  ic_method?: string;
  weighting_scheme?: string;
  quantile?: number | null;
  commission_bps?: number;
  slippage_bps?: number;
}

export interface RunBacktestMetrics {
  signed_ic: number | null;
  mean_abs_ic: number | null;
  ic?: number | null;
  ic_ir: number | null;
  gross_sharpe: number | null;
  net_sharpe: number | null;
  max_drawdown: number | null;
  turnover: number | null;
  avg_gross: number | null;
  avg_positions: number | null;
  max_position: number | null;
  usable: boolean;
}

export interface RunBacktestReport {
  start: string | null;
  end: string | null;
  observations: number;
  metrics: RunBacktestMetrics;
  returns: FormulaTestReturnPoint[];
  normalized_equity: FormulaTestEquityPoint[];
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
  enabled_categories?: string[] | null;
}

export type TrainingResourceProfile = "light" | "auto" | "maximum" | "custom";

export interface TrainingResourcesRequest {
  profile: TrainingResourceProfile;
  cpu_budget_percent?: number | null;
}

export interface ResolvedTrainingResources extends TrainingResourcesRequest {
  percent: number;
  detected_cpus: number;
  worker_capacity: number;
  requested_workers: number;
  effective_workers: number;
  /** Backward-compatible alias of effective_workers. */
  workers: number;
  available_memory_bytes: number;
  memory_budget_bytes: number;
  memory_per_worker_bytes: number;
  run_memory_budget_bytes: number;
  accelerated: boolean;
  fallback_reason: string | null;
}

export interface TrainingCapabilities {
  default_profile: "auto";
  detected_cpus: number;
  worker_capacity: number;
  available_memory_bytes: number;
  memory_budget_bytes: number;
  cpu_budget_percent_min: number;
  cpu_budget_percent_max: number;
  profiles: Record<"light" | "auto" | "maximum", ResolvedTrainingResources>;
  evaluator: "auto" | "python" | "cpp";
  cpp_available: boolean;
  fallback_reason: string | null;
}

export interface ProgressSnapshot {
  phase: string;
  generation: number;
  target_generations: number;
  history: Array<{ generation: number; best_fitness: number; mean_fitness: number }>;
  best: { tree: string; fitness: number } | null;
  resources?: ResolvedTrainingResources | null;
  candidate_done?: number;
  candidate_total?: number;
  report_done?: number;
  report_total?: number;
  factors_per_second?: number | null;
  termination_reason?: string | null;
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
  resources?: ResolvedTrainingResources | null;
  termination_reason?: string;
  report_cancelled?: boolean;
}

export interface SessionJob {
  id: string;
  status: string; // queued | running | done | stopped | failed
  progress: ProgressSnapshot | null;
  termination_reason?: string | null;
}

export interface SessionState {
  id: string;
  name: string;
  created_at: string;
  universe: string;
  as_of: string;
  boundaries: SessionBoundaries;
  config: Partial<GpConfig>;
  resources?: TrainingResourcesRequest;
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
  as_of: string;
  config?: Partial<GpConfig>;
  operators?: OperatorSpec[];
  seed_factor_ids?: string[];
  train?: number;
  valid?: number;
  embargo?: number;
  resources?: TrainingResourcesRequest;
}

export interface SessionContinueRequest {
  generations: number;
  config?: Partial<GpConfig>;
  universe?: string | null;
  operators?: OperatorSpec[];
  seed_factor_ids?: string[];
  resources?: TrainingResourcesRequest;
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
  metrics: Record<string, number | boolean | null>;
  provenance: FactorProvenance;
  required_operators: OperatorSpec[];
  expanded_tree?: FactorNode;
  notes: string;
  disclaimer: string;
}

export type FormulaResultKind = "training" | "backtest";

export interface FormulaResultSeriesPoint {
  date: string;
  gross_return: number | null;
  net_return: number | null;
  equity: number | null;
}

/** Canonical user-facing result. SavedFactor remains the legacy wire alias. */
export interface FormulaResult extends SavedFactor {
  kind: FormulaResultKind;
  out_type?: string;
  source_formula?: string | null;
  dependency_fingerprint?: string | null;
  bindings?: Record<string, FormulaTestBinding>;
  source?: FormulaTestSource | null;
  expression_fingerprint?: string | null;
  universe?: string | null;
  start?: string | null;
  end?: string | null;
  horizon?: number | null;
  weighting_scheme?: "quantile_ls" | "rank_proportional" | null;
  quantile?: number | null;
  commission_bps?: number | null;
  slippage_bps?: number | null;
  data_coverage?: FormulaTestCoverage | null;
  data_revision?: string | null;
  returns?: FormulaTestReturnPoint[];
  normalized_equity?: FormulaTestEquityPoint[];
  series?: FormulaResultSeriesPoint[];
}

export type FormulaTestBinding =
  | { kind: "field"; field: string }
  | { kind: "formula"; runtime_name: string }
  | { kind: "result"; result_id: string }
  | { kind: "literal"; value: number };

export type FormulaTestSource =
  | {
      kind: "draft";
      body: FactorNode;
      inputs: FormulaInputSpec[];
      out_type: string;
    }
  | { kind: "saved"; runtime_name: string };

export interface FormulaTestRequest {
  source: FormulaTestSource;
  /** Keys are exposed-input names. */
  bindings: Record<string, FormulaTestBinding>;
  universe?: string;
  start?: string | null;
  end?: string | null;
  horizon?: number;
  weighting_scheme?: "quantile_ls" | "rank_proportional";
  quantile?: number;
  commission_bps?: number;
  slippage_bps?: number;
}

export interface FormulaTestCoverage {
  first_date: string | null;
  last_date: string | null;
  observations: number;
  symbols: number;
  warnings: string[];
}

export interface FormulaTestMetrics {
  ic?: number | null;
  ic_ir?: number | null;
  gross_sharpe: number | null;
  net_sharpe: number | null;
  max_drawdown: number | null;
  turnover: number | null;
  avg_gross?: number | null;
  avg_positions: number | null;
  max_position: number | null;
  usable: boolean;
}

export interface FormulaTestReturnPoint {
  date: string;
  gross: number | null;
  net: number | null;
}

export interface FormulaTestEquityPoint {
  date: string;
  value: number | null;
}

export interface FormulaTestResultPayload {
  kind: "backtest";
  tree: FactorNode;
  expanded_tree?: FactorNode;
  dependency_revisions: string[];
  expression_fingerprint: string;
  source: FormulaTestSource;
  bindings: Record<string, FormulaTestBinding>;
  universe: string;
  start: string | null;
  end: string | null;
  horizon: number;
  weighting_scheme: "quantile_ls" | "rank_proportional";
  quantile: number | null;
  commission_bps: number;
  slippage_bps: number;
  data_coverage: FormulaTestCoverage;
  data_revision: string;
  metrics: FormulaTestMetrics;
  returns: FormulaTestReturnPoint[];
  normalized_equity: FormulaTestEquityPoint[];
  disclaimer: string;
}

export interface FormulaTestJob {
  job_id: string;
  status: "queued" | "running" | "done" | "stopped" | "failed";
  result?: FormulaTestResultPayload | null;
  error?: string | null;
  progress?: { phase: string } | null;
  termination_reason?: string | null;
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
  logical_name?: string;
  display_name?: string;
  description?: string;
  kind: string; // operator | operand | ephemeral
  arg_types: string[];
  inputs?: FormulaInputSpec[];
  out_type: string;
  user: boolean;
  origin?: "builtin" | "user_formula" | "catalog_formula" | "data" | "value";
  editable?: boolean;
  category?: string;
  revision?: number | null;
  runtime_name?: string;
  family?: string;
  aliases?: string[];
  catalog_revision?: number | null;
}

export interface OperatorSpec {
  name: string;
  arg_types: string[];
  out_type: string;
  body: FactorNode; // a typed body tree with $arg leaves
}

export interface FormulaSpec {
  name: string;
  display_name: string;
  description: string;
  arg_types: string[];
  inputs?: FormulaInputSpec[];
  out_type: string;
  body: FactorNode;
  category?: string;
  registered?: boolean;
  error?: string | null;
  revision?: number;
  runtime_name?: string;
  created_at?: string;
  updated_at?: string;
  origin?: "user_formula" | "catalog_formula";
  editable?: boolean;
  family?: string;
  aliases?: string[];
  catalog_revision?: number | null;
}

export interface FormulaInputSpec {
  name: string;
  type: string;
  description: string;
  /** Optional, visible call-site default for scalar/window inputs only. */
  default?: number | null;
}

export interface FormulaImpact {
  name: string;
  runtime_name: string;
  change: "none" | "metadata" | "calculation";
  direct_formulas: string[];
  transitive_formulas: string[];
  factors: string[];
  sessions: string[];
  /** Active one-shot training jobs that pinned a revision from this family. */
  runs?: string[];
  has_references: boolean;
}

export interface FormulaDetail extends FormulaSpec {
  revisions: FormulaSpec[];
  impact: FormulaImpact;
}

export interface FormulaValidation {
  ok: boolean;
  out_type?: string;
  name?: string | null;
  error?: string | null;
}

export interface CategorySettings {
  order: string[];
  overrides: Record<string, string>;
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

export type UniverseMode = "static_snapshot" | "point_in_time";
export type UniverseSource = "bundled" | "sample" | "custom" | "bundled_static";

export interface UniverseDefinition {
  id: string;
  display_name: string;
  snapshot_date?: string | null;
  interval_semantics: string;
  member_count: number;
}

export interface UniverseProvenance {
  provider: string;
  source_url?: string | null;
  retrieved_at?: string | null;
  attribution?: string | null;
  license?: string | null;
  license_url?: string | null;
  terms_url?: string | null;
  note?: string | null;
}

export interface UniverseReadiness {
  membership_ready: boolean;
  price_ready: boolean | null;
  training_ready?: boolean | null;
  research_ready: boolean;
  coverage_checked?: boolean;
  issues: string[];
}

export interface UniverseCacheCoverage {
  as_of: string;
  eligible_symbols: string[];
  cached_symbols: string[];
  missing_symbols: string[];
  invalid_symbols?: string[];
  uncovered_symbols?: string[];
  late_start_symbols?: string[];
  stale_symbols?: string[];
  incomplete_symbols?: string[];
  symbol_coverage?: Record<string, {
    first_date: string | null;
    last_date: string | null;
    issues: string[];
  }>;
  complete: boolean;
}

export interface UniverseInfo extends UniverseSpec {
  display_name?: string;
  symbols: string[];
  membership_count?: number;
  symbol_count?: number;
  source: UniverseSource;
  mode?: UniverseMode;
  definition?: UniverseDefinition;
  fingerprint?: string;
  provenance?: UniverseProvenance;
  readiness?: UniverseReadiness;
  /** Canonical definition symbol -> provider/cache symbol. */
  aliases?: Record<string, string>;
  integrity?: UniverseIntegrity;
  cache_coverage?: UniverseCacheCoverage;
}

export interface UniverseIntegrity {
  membership_history: string;
  coverage: string;
  research_ready: boolean;
  provenance: string;
  warning: string;
}

export interface UniversePreset extends UniverseIntegrity {
  id: string;
  display_name: string;
  benchmark: string;
  status: "bundled_sample" | "import_required" | "custom_import" | string;
  available: boolean;
  snapshot_universe?: string | null;
  snapshot_available?: boolean;
  pit_import_name?: string | null;
  pit_available?: boolean;
  pit_universe?: string | null;
  mode?: UniverseMode;
  definition?: UniverseDefinition;
  fingerprint?: string;
  provenance_detail?: UniverseProvenance;
  readiness?: UniverseReadiness;
  aliases?: Record<string, string>;
}

export interface UniverseDraft {
  name: string;
  rows: Array<{ symbol: string; entry: string; exit: string }>;
  selectedUniverse?: string;
  expectedStart?: string;
}

export interface FormulaDraft {
  name: string;
  display_name: string;
  description: string;
  template?: string;
  arg_types?: string[];
  out_type?: string;
  body?: FactorNode;
  inputs?: FormulaInputSpec[];
  category?: string;
  expression?: string;
  activeMode?: "visual" | "expression";
  loadedName?: string | null;
  loadedRevision?: number | null;
  graphNodes?: FormulaDraftNode[];
  graphEdges?: FormulaDraftEdge[];
  selectedNodeId?: string | null;
  /** Read-only migration data written by releases that had two builder modes. */
  builderDrafts?: Partial<Record<"factor" | "reusable", FormulaDraft>>;
  recoveredDrafts?: Array<{ label: string; draft: FormulaDraft }>;
}

export interface FormulaDraftNode {
  id: string;
  type: string;
  x: number;
  y: number;
  data: Record<string, unknown>;
}

export interface FormulaDraftEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
}

export interface SymbolCandidate {
  symbol: string;
  name: string;
  exchange: string;
  quote_type: string;
  currency: string;
  source: string;
}

export interface SymbolValidation {
  symbol: string;
  valid: boolean;
  rows: number;
  first_date?: string | null;
  last_date?: string | null;
  provider?: string | null;
  error?: string | null;
  cached?: boolean;
}

export interface DataCoverage {
  symbol: string;
  cached: boolean;
  rows: number;
  first_date?: string | null;
  last_date?: string | null;
  requested_start?: string | null;
  requested_end?: string | null;
  needs_sync: boolean;
}

export interface DataSyncRequest {
  /** Prefer a saved universe so memberships and aliases are resolved atomically. */
  universe?: string;
  /** Compatibility path for a draft or one-off symbol pull. */
  symbols?: string[];
  start: string;
  end?: string | null;
  mode: "incremental" | "refresh";
}

export interface DataSyncResult {
  symbol: string;
  status: "queued" | "running" | "done" | "failed" | "fetched" | "skipped";
  rows_fetched: number;
  rows_cached: number;
  first_date?: string | null;
  last_date?: string | null;
  provider?: string | null;
  provider_symbol?: string | null;
  error?: string | null;
}

export interface SyncProgressSnapshot {
  done: number;
  total: number;
  current_symbol: string | null;
}

export interface DataSyncJob {
  job_id: string;
  status: "queued" | "running" | "stopping" | "stopped" | "done" | "failed";
  stopping?: boolean;
  termination_reason?: "completed" | "user_stopped" | string | null;
  request?: DataSyncRequest & {
    resolved_symbols?: string[];
    aliases?: Record<string, string>;
  };
  universe_definition?: {
    name: string;
    requested_name: string;
    source: UniverseSource;
    mode: UniverseMode;
    definition: UniverseDefinition;
    fingerprint: string;
    provenance: UniverseProvenance;
    aliases: Record<string, string>;
  } | null;
  result: {
    mode: "incremental" | "refresh";
    start: string;
    end?: string | null;
    universe?: string | null;
    resolved_symbols?: string[];
    aliases?: Record<string, string>;
    universe_definition?: DataSyncJob["universe_definition"];
    termination_reason?: "completed" | "user_stopped" | string;
    results: DataSyncResult[];
  } | null;
  error: string | null;
  progress?: SyncProgressSnapshot | null;
}

export interface MembershipSyncRequest {
  symbols: string[];
  expected_start: string;
}

export interface MembershipSyncResult {
  symbol: string;
  status: "resolved" | "failed";
  entry?: string | null;
  exit?: string | null;
  delisted: boolean;
  list_date?: string | null;
  last_date?: string | null;
  error?: string | null;
}

export interface MembershipSyncJob {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  result: {
    expected_start: string;
    results: MembershipSyncResult[];
  } | null;
  error: string | null;
  progress?: SyncProgressSnapshot | null;
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
  formulaDraft?: FormulaDraft;
  operatorDraft?: OperatorComposerDraft;
  ui: WorkspaceUiState;
}

export interface WorkspaceSummary {
  id: string;
  name: string;
  savedAt: string;
  hasRun: boolean;
}
