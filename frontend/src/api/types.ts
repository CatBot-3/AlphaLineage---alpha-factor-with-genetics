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
  validation_passed?: boolean;
  validation_reason?: string | null;
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
  best_signed_ic?: number;
  unique_tree_count?: number;
  unique_tree_ratio?: number;
  duplicate_count?: number;
  novel_offspring_count?: number;
  champion_age?: number;
  generations_since_improvement?: number;
  diversity_warning?: number | boolean;
  parameter_neighbor_count?: number;
  formula_default_injection_count?: number;
  formula_exploration?: Record<string, FormulaParameterCoverage>;
}

export interface CandidateScoreMetrics {
  ic?: number;
  signed_ic?: number;
  oriented_ic?: number;
  mean_abs_ic?: number;
  ic_ir?: number;
  oriented_ic_ir?: number;
  polarity?: number;
  sign_consistency?: number;
  valid_dates?: number;
  valid_date_coverage?: number;
  avg_active_names?: number;
  min_active_names?: number;
  varying_factor_coverage?: number;
  exposure_coverage?: number;
  two_sided_coverage?: number;
}

export interface ValidationSelection {
  validated: boolean;
  reason?: string | null;
  first_seen: number;
  polarity: 1 | -1;
  training_fitness: number;
  training_metrics: CandidateScoreMetrics;
  validation_fitness: number;
  validation_metrics: CandidateScoreMetrics;
  source_expression?: string;
  oriented_expression?: string;
  folds?: ValidationFold[];
  positive_folds?: number;
  required_positive_folds?: number;
  median_oriented_ic?: number | null;
  worst_fold_ic?: number | null;
  complexity?: ValidationComplexity;
  final_objective?: number | null;
  parameter_variants?: number;
  parameter_coverage?: Record<string, FormulaParameterCoverage>;
}

export interface ValidationFold {
  index: number;
  start?: string | null;
  end?: string | null;
  observations?: number;
  valid_dates: number;
  ic_coverage?: number | null;
  signed_ic?: number | null;
  oriented_ic?: number | null;
  ic_ir?: number | null;
  avg_active_names?: number | null;
  min_active_names?: number | null;
  varying_factor_dates?: number;
  varying_factor_coverage?: number | null;
  active_weight_dates?: number;
  exposure_coverage?: number | null;
  two_sided_dates?: number;
  two_sided_coverage?: number | null;
  realized_coverage?: number | null;
  avg_gross_exposure?: number | null;
  adequate_coverage?: boolean;
  coverage_failures?: string[];
  positive: boolean;
}

export interface ValidationComplexity {
  expanded_nodes: number;
  mode: "per_node" | "normalized_budget";
  penalty_value: number;
  deduction: number;
  max_nodes: number;
}

export interface FormulaParameterCoverage {
  calls_searched?: number;
  distinct_parameter_tuples?: number;
  best_training_score?: number | null;
  best_validation_score?: number | null;
}

export interface RunResult {
  best_factor: string | FactorNode; // the service returns a JSON string; demo JSON may store a tree
  /** Null for validation-only rounds until the user explicitly finalizes one. */
  report: Report | null;
  generations: number;
  history: HistoryPoint[];
  lineage: Lineage;
  // session-aware fields (present for runs launched as a session segment)
  session_id?: string;
  segment?: number;
  round_index?: number;
  round_metadata?: SessionRoundSummary;
  test_reads?: number;
  cumulative_trials?: number;
  repeated_oos_warning?: boolean;
  test_read_index?: number | null;
  evidence_status?: EvidenceStatus;
  validity?: "valid" | "invalid_legacy_semantics";
  restart_required?: boolean;
  selection?: ValidationSelection;
  formula_revisions?: Array<{ runtime_name?: string; name?: string; revision?: number }>;
  resources?: ResolvedTrainingResources | null;
  termination_reason?: "completed" | "user_stopped" | "time_budget";
  timings?: {
    training_seconds: number;
    reporting_seconds: number;
    total_seconds: number;
  };
  context?: RunContext;
  oos_backtest?: RunBacktestReport | null;
  strategy_results?: StrategyBacktestResult[];
  primary_strategy_id?: string | null;
  strategy_plan_id?: string | null;
  comparison_id?: string | null;
  validation_only?: boolean;
  finalization?: SessionFinalizationDetail | null;
  session_holdout_reads?: number;
  inherited_test_reads?: number;
  inherited_evidence_sources?: string[];
  holdout_fingerprint?: string | null;
  same_holdout_read_index?: number | null;
  parameter_coverage?: Record<string, FormulaParameterCoverage>;
}

/**
 * When a signal computed from day t data (including the t close) is traded.
 * `close` is the legacy same-close assumption; stored sessions without the key mean `close`.
 */
export type ExecutionTiming = "close" | "next_open" | "next_close";

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
  execution?: ExecutionTiming;
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
  insolvent?: boolean;
  missing_return_observations?: number;
}

export interface RunBacktestReport {
  start: string | null;
  end: string | null;
  observations: number;
  metrics: RunBacktestMetrics;
  returns: FormulaTestReturnPoint[];
  normalized_equity: FormulaTestEquityPoint[];
  portfolio_health?: PortfolioHealth;
  integrity?: {
    valid: boolean;
    issues: string[];
    equity_terminated_reason?: "missing_realized_return" | "insolvent" | string | null;
  };
}

export interface PortfolioHealth {
  eligible_dates: number;
  calendar_observations: number;
  active_observations: number;
  two_sided_observations: number;
  exposure_coverage: number;
  flat_factor_dates: number;
  valid: boolean;
  reason: string | null;
}

export interface PortfolioStrategySpec {
  id: string;
  scheme: "quantile_ls" | "rank_proportional";
  quantile?: number | null;
}

export interface StrategyBacktestResult {
  strategy_id: string;
  spec: PortfolioStrategySpec;
  role?: "primary" | "comparison";
  /** Present for pre-holdout strategy comparisons. */
  validation_backtest?: RunBacktestReport | null;
  /** Present only after the frozen strategy bundle is explicitly finalized. */
  oos_backtest?: RunBacktestReport | null;
  folds?: Array<Record<string, unknown>>;
  eligible?: boolean;
}

export interface BenchmarkDefinition {
  id: "sp500" | "djia" | "nasdaq100" | string;
  label: string;
  symbol: string;
  color: string;
  return_type: "price_return";
  methodology: string;
}

export interface BenchmarkSeries extends BenchmarkDefinition {
  status: "ready" | "partial" | "needs_sync";
  message: string | null;
  requested_start: string;
  requested_end: string;
  coverage: DataCoverage;
  normalized_equity: FormulaTestEquityPoint[];
  sync_request: DataSyncRequest;
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
  /**
   * Legacy/direct callers remain per-node. The app explicitly requests a
   * normalized total deduction so larger legal formulas are not priced out.
   */
  complexity_penalty_mode?: "per_node" | "normalized_budget";
  complexity_penalty_value?: number | null;
  parameter_neighbor_fraction?: number;
  validation_folds?: number;
  /**
   * Missing legacy requests retain the classic trajectory. The app explicitly
   * opts new sessions into protected valley-crossing exploration.
   */
  exploration_profile?: "classic" | "balanced" | "aggressive";
  elitism: number;
  ic_method: string;
  min_names: number;
  horizon: number;
  /** Frozen when a session is created. Missing on legacy sessions, which mean `close`. */
  execution?: ExecutionTiming;
  min_depth: number;
  seed: number;
  time_budget_s: number | null;
  enabled_categories?: string[] | null;
  /** Logical formula allow-list. Null preserves the legacy category-wide selection. */
  enabled_formula_names?: string[] | null;
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
  requested_generations?: number;
  started_at?: string;
  completed_at?: string | null;
  report_available?: boolean;
  test_read_index?: number | null;
  evidence_status?: EvidenceStatus;
  selected_lineage_node_id?: number | null;
  timings?: RunResult["timings"];
  panel_fingerprint?: string;
  scorer_version?: number;
  evolution_version?: number;
  adjustment_version?: number;
}

export interface SessionRoundSummary {
  index: number;
  segment_index: number;
  /** Present on rounds the agent authored; absent on GP segments. */
  origin?: "agent";
  agent_expression?: string;
  parent_round_index?: number | null;
  report_available: boolean;
  test_read_index: number | null;
  evidence_status: EvidenceStatus;
  status: string;
  started_at?: string | null;
  completed_at?: string | null;
  requested_generations: number;
  gen_start: number;
  gen_end: number;
  termination_reason?: string | null;
  selected_lineage_node_id?: number | null;
  resources?: ResolvedTrainingResources | null;
  timings?: RunResult["timings"];
  panel_fingerprint?: string | null;
  scorer_version?: number | null;
  evolution_version?: number | null;
  adjustment_version?: number | null;
  validity?: "valid" | "invalid_legacy_semantics";
  restart_required?: boolean;
  invalid_reason?: string;
  finalization_available?: boolean;
  latest_finalization_id?: string | null;
  latest_strategy_comparison_id?: string | null;
  latest_strategy_plan_id?: string | null;
}

export type EvidenceStatus =
  | "validation_only"
  | "post_holdout_adaptive"
  | "locked_first_read"
  | "repeated_same_holdout"
  // Legacy aliases retained for old saved sessions/demo payloads.
  | "locked"
  | "exploratory_repeat";

export interface SessionFinalizationSummary {
  evaluation_id: string;
  round_index: number;
  status: string;
  evidence_status: "locked_first_read" | "repeated_same_holdout";
  same_holdout_read_index: number;
  session_holdout_reads: number;
  holdout_fingerprint: string;
  started_at?: string | null;
  completed_at?: string | null;
  report_available: boolean;
  strategy_plan_id?: string | null;
  comparison_id?: string | null;
  primary_strategy_id?: string | null;
  inherited_evidence_sources?: string[];
  /** Deprecated compatibility aggregate. */
  test_reads?: number;
}

export interface SessionFinalizationDetail extends SessionFinalizationSummary {
  best_factor: string | FactorNode;
  report: Report;
  oos_backtest?: RunBacktestReport | null;
  strategy_results?: StrategyBacktestResult[];
  context?: RunContext;
  selection?: ValidationSelection;
}

export interface SessionFinalizationHandle {
  session_id: string;
  round_index: number;
  evaluation_id: string;
  job_id: string;
  status: string;
}

export interface SessionJob {
  id: string;
  job_id?: string;
  status: string; // queued | running | done | stopped | failed
  progress: ProgressSnapshot | null;
  termination_reason?: string | null;
  metadata?: Record<string, unknown>;
  error?: string | null;
  evaluation_id?: string;
  round_index?: number;
  strategy_plan_id?: string | null;
  comparison_id?: string | null;
  primary_strategy_id?: string | null;
}

export interface StrategyComparisonSummary {
  comparison_id: string;
  round_index: number;
  status: string;
  evidence_status?: "validation_only" | "post_holdout_adaptive" | string;
  prior_holdout_reads?: number;
  created_at?: string | null;
  completed_at?: string | null;
  strategies?: PortfolioStrategySpec[];
  strategy_ids?: string[];
  result_available?: boolean;
  job_id?: string | null;
  error?: string | null;
}

export interface StrategyComparisonDetail extends StrategyComparisonSummary {
  session_id?: string;
  schema_version?: number;
  portfolio_schema_version?: string;
  costs?: { commission_bps?: number; slippage_bps?: number };
  strategy_results: StrategyBacktestResult[];
}

export interface StrategyComparisonHandle {
  session_id: string;
  round_index: number;
  comparison_id: string;
  job_id: string;
  status: string;
}

export interface FinalizationPlan {
  strategy_plan_id: string;
  session_id?: string;
  round_index: number;
  comparison_id: string;
  primary_strategy_id: string;
  strategies?: PortfolioStrategySpec[];
  strategy_ids?: string[];
  costs?: { commission_bps?: number; slippage_bps?: number };
  holdout_fingerprint?: string | null;
  source_comparison_evidence_status?: "validation_only" | "post_holdout_adaptive" | string;
  created_at?: string | null;
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
  session_holdout_reads?: number;
  inherited_test_reads?: number;
  inherited_evidence_sources?: string[];
  holdout_read_counts?: Record<string, number>;
  segments: SessionSegment[];
  rounds?: SessionRoundSummary[];
  last_job_id: string | null;
  job: SessionJob | null;
  finalization_job?: SessionJob | null;
  last_finalization_job?: SessionJob | null;
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
  updated_at?: string;
  current_generation?: number;
  last_status?: string | null;
  has_checkpoint?: boolean;
  has_report?: boolean;
  latest_completed_round?: number | null;
  requested_generations?: number | null;
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
  execution?: ExecutionTiming | null;
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
  execution?: ExecutionTiming;
  strategies?: PortfolioStrategySpec[];
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
  /** Date on which the factor was observed and the newest cohort was formed. */
  signal_date?: string;
  /** Date on which the close-to-close portfolio return was realized. */
  date: string;
  gross: number | null;
  net: number | null;
  active?: boolean;
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
  oos_backtest?: RunBacktestReport;
  portfolio_health?: PortfolioHealth;
  strategy_results?: StrategyBacktestResult[];
  primary_strategy_id?: string;
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
  /**
   * Additive backend metadata. Optional so the packaged frontend remains
   * compatible with an older local backend during rolling upgrades.
   */
  tiingo_api_key_source?: "environment" | "stored" | "none";
  tiingo_stored_key_set?: boolean;
  evaluator: "auto" | "python" | "cpp";
  cpp_available: boolean;
  /** The agent's model. Optional for the same rolling-upgrade reason. */
  llm_provider?: string;
  llm_model?: string;
  llm_base_url?: string;
  llm_api_key_set?: boolean;
  llm_api_key_source?: "environment" | "stored" | "none";
  llm_configured?: boolean;
}

export interface SettingsUpdate {
  factors_dir?: string;
  tiingo_api_key?: string;
  evaluator?: "auto" | "python" | "cpp";
  llm_provider?: string;
  llm_model?: string;
  llm_base_url?: string;
  /** An empty string clears the stored key, matching the Tiingo contract. */
  llm_api_key?: string;
  llm_api_key_provider?: string;
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
  constraints?: FormulaConstraint[];
  status?: "active" | "retired";
  replacement?: string | null;
  family_order?: number | null;
}

export interface OperatorSpec {
  name: string;
  arg_types: string[];
  out_type: string;
  body: FactorNode; // a typed body tree with $arg leaves
}

export interface FormulaConstraint {
  left: string;
  operator: "lt" | "le" | "gt" | "ge" | "ne";
  right: string;
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
  constraints?: FormulaConstraint[];
  status?: "active" | "retired";
  replacement?: string | null;
  family_order?: number | null;
}

export interface FormulaInputSpec {
  name: string;
  type: string;
  description: string;
  /** Optional, visible call-site default for scalar/window inputs only. */
  default?: number | null;
  role?: "data" | "parameter";
  tuning?: {
    enabled: boolean;
    min: number;
    max: number;
    step: number;
    radius: number;
  } | null;
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
  /** Earliest membership date that must be cached for point-in-time integrity. */
  required_start?: string | null;
  /** Latest date included in this readiness decision. */
  required_end?: string | null;
  eligible_symbols: string[];
  /** Memberships still active at the coverage cutoff (static snapshots are included here). */
  active_symbols?: string[];
  /** Symbols whose declared membership exit is on or before the coverage cutoff. */
  exited_symbols?: string[];
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
    required_start?: string | null;
    required_end?: string | null;
    membership_status?: "active" | "exited" | "current_snapshot" | string;
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
  /** Presentation folder (`null` = top level). Folders never change the definition itself. */
  folder_id?: string | null;
  /** Folder names from the top level down; lets pickers group without a second request. */
  folder_path?: string[];
}

export interface UniverseFolder {
  id: string;
  name: string;
  parent: string | null;
  builtin: boolean;
}

export interface UniverseFolderTree {
  folders: UniverseFolder[];
  /** Universe name -> folder id (`null` = top level). */
  placements: Record<string, string | null>;
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
  /** Canonical cache symbol -> provider symbol; normally resolved from a saved universe. */
  aliases?: Record<string, string>;
}

export interface DataSyncResult {
  symbol: string;
  status: "queued" | "running" | "done" | "failed" | "fetched" | "skipped" | "quota_exceeded";
  rows_fetched: number;
  rows_cached: number;
  first_date?: string | null;
  last_date?: string | null;
  provider?: string | null;
  provider_symbol?: string | null;
  error?: string | null;
  /** Set when the provider refused the request because an allowance window is used up. */
  quota_scope?: "hourly" | "daily" | "monthly" | "unknown" | string | null;
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
    failed_count?: number;
    succeeded_count?: number;
    termination_reason?: "completed" | "user_stopped" | "quota_exceeded" | string;
    results: DataSyncResult[];
    /** Present when the batch stopped because the provider allowance was exhausted. */
    quota?: { provider: string; scope: string; message: string } | null;
    /** Symbols the stopped batch never requested; re-run the sync after the window resets. */
    not_attempted?: string[];
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
  status: "resolved" | "unverified_stale" | "failed";
  /** Deprecated and always inert: price coverage cannot establish membership. */
  entry?: string | null;
  /** Deprecated and always inert: an absent price tail cannot establish an exit. */
  exit?: string | null;
  /** Deprecated and always false without an authoritative security-status source. */
  delisted: boolean;
  review_needed?: boolean;
  first_date?: string | null;
  /** Deprecated alias for first_date. */
  list_date?: string | null;
  last_date?: string | null;
  note?: string | null;
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
  selectedTab?: "train" | "dashboard" | "factor" | "genealogy" | "extend" | "library" | "agent";
  selectedFactorNode?: { name: string; value?: number } | null;
  selectedLineage?: number | null;
  sessionId?: string | null;
  selectedRound?: number | null;
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

// --- P11: the agent -----------------------------------------------------------
export type ToolSafety = "read" | "evaluate" | "propose" | "annotate";

export interface AgentToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  safety: ToolSafety;
}

export interface AgentToolCatalog {
  tools: AgentToolSpec[];
  tunable_config_keys: string[];
  protected_config_keys: Record<string, string>;
  defaults: { max_tool_calls: number; max_evaluations: number; max_seconds: number };
  unavailable_reason: string;
  disclaimer: string;
}

export interface AgentToolCall {
  tool: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
  safety: ToolSafety | "unknown";
  ok: boolean;
}

export interface AgentLabel {
  name: string;
  evidence?: string;
  reading?: string;
  risk?: string;
}

export interface AgentProposal {
  id: string;
  kind: "factor" | "config";
  rationale: string;
  payload: {
    name?: string;
    expression?: string;
    tree?: FactorNode;
    patch?: Record<string, unknown>;
    diff?: { key: string; from: unknown; to: unknown }[];
  };
  applied: boolean;
  round_index?: number;
}

export interface AgentTurn {
  role: "user" | "assistant";
  text: string;
  at: string;
  calls: AgentToolCall[];
  labels: AgentLabel[];
  proposals: AgentProposal[];
  usage: Record<string, unknown>;
  stop_reason: string;
  error: string;
}

export interface Conversation {
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  turns: AgentTurn[];
  summary: string;
  summarised_through: number;
  disclaimer: string;
}

export interface AgentMessageRequest {
  message: string;
  round_index?: number;
  provider?: string;
  model?: string;
  base_url?: string;
  budget?: { max_tool_calls: number; max_evaluations: number; max_seconds: number };
}

export interface AgentJob {
  job_id: string;
  status: "queued" | "running" | "done" | "failed" | "stopped";
  error: string | null;
  result: Record<string, unknown> | null;
}

// --- overlap with known factors ------------------------------------------------------------
export type OverlapVerdict =
  | "novel"
  | "related"
  | "mostly_explained"
  | "near_duplicate"
  | "unmeasured";

export interface OverlapReference {
  key: string;
  name: string;
  display_name: string;
  /** catalog = starter formulas, your_formulas = user formulas, saved_results = Formula Results. */
  group: "catalog" | "your_formulas" | "saved_results" | string;
  family: string;
  status: "ok" | "unavailable" | "no_overlap";
  error?: string;
  /** Mean over training dates of the cross-sectional Spearman correlation (signed). */
  mean_rank_corr?: number | null;
  abs_mean_rank_corr?: number | null;
  share_dates_abs_corr_above_half?: number | null;
  dates?: number;
  reference_ic?: number | null;
  reference_ic_t?: number | null;
  /** Set when this factor was not removed because it nearly copies one that was. */
  redundant_with?: string;
}

export interface FactorOverlapReport {
  overlap_version: number;
  session_id: string;
  round_index: number;
  window: { start: string | null; end: string | null; dates: number; label: string };
  thresholds: { related: number; near_duplicate: number; top_k: number };
  candidate: { ic: number | null; ic_t: number | null };
  residual: {
    /** Unique IC: the additive part of the IC the known factors cannot account for. */
    ic: number | null;
    ic_t: number | null;
    explained_ic: number | null;
    decomposed_ic: number | null;
    unique_share: number | null;
    mean_r_squared: number | null;
    explained_by: string[];
  };
  max_abs_rank_corr: number | null;
  verdict: OverlapVerdict;
  references: OverlapReference[];
  reference_count: number;
  measured_count: number;
  horizon: number;
  execution: ExecutionTiming;
  include_saved_results: boolean;
  computed_at: string;
  disclaimer?: string;
  stale?: boolean;
  stale_reasons?: string[];
}
