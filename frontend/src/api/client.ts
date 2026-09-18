// Typed client for the `app` build: submit a GP run and poll it to completion.

import type {
  AgentJob,
  Conversation,
  AgentMessageRequest,
  AgentToolCatalog,
  BenchmarkDefinition,
  BenchmarkSeries,
  CategorySettings,
  DataCoverage,
  DataSyncJob,
  DataSyncRequest,
  DataUsageRow,
  FormulaSpec,
  FormulaDetail,
  FormulaImpact,
  FormulaResult,
  FormulaValidation,
  FormulaTestJob,
  FormulaTestRequest,
  FactorOverlapReport,
  FinalizationPlan,
  Lineage,
  MembershipSyncJob,
  MembershipSyncRequest,
  OperatorSpec,
  RunResult,
  PrimitiveInfo,
  ProgressSnapshot,
  SavedFactor,
  SessionContinueRequest,
  SessionCreateRequest,
  SessionFinalizationDetail,
  SessionFinalizationHandle,
  SessionFinalizationSummary,
  SessionState,
  SessionSummary,
  SessionRoundSummary,
  Settings,
  SettingsUpdate,
  SymbolCandidate,
  SymbolValidation,
  TrainingCapabilities,
  PortfolioStrategySpec,
  StrategyComparisonDetail,
  StrategyComparisonHandle,
  StrategyComparisonSummary,
  UniverseFolder,
  UniverseFolderTree,
  UniverseInfo,
  UniverseCacheCoverage,
  UniversePreset,
  UniverseSpec,
  WorkspaceSnapshot,
  WorkspaceSummary,
} from "./types";

// Same-origin by default (the Docker image serves the UI from the API host); `.env.app`
// sets an explicit base for the Vite dev server, which runs on a different port.
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function jsonOrThrow<T>(res: Response, action: string): Promise<T> {
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new ApiError(detail.detail ?? `${action} failed: ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

const POST = (path: string, body: unknown): Promise<Response> =>
  fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export interface RunConfig {
  population_size?: number;
  generations?: number;
  seed?: number;
  [key: string]: unknown;
}

interface JobResponse {
  job_id: string;
  status: string;
}

interface RunStatus {
  job_id: string;
  status: string;
  result: RunResult | null;
  error: string | null;
  progress?: ProgressSnapshot | null;
}

export async function submitRun(config: RunConfig = {}): Promise<string> {
  const res = await fetch(`${BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!res.ok) throw new Error(`submit failed: ${res.status}`);
  return ((await res.json()) as JobResponse).job_id;
}

export async function getRun(jobId: string): Promise<RunStatus> {
  const res = await fetch(`${BASE}/runs/${jobId}`);
  if (!res.ok) throw new Error(`status failed: ${res.status}`);
  return (await res.json()) as RunStatus;
}

export async function listBenchmarks(): Promise<BenchmarkDefinition[]> {
  return jsonOrThrow(await fetch(`${BASE}/benchmarks`), "list benchmarks");
}

export async function getBenchmarkSeries(
  benchmarkId: string,
  start: string,
  end: string,
): Promise<BenchmarkSeries> {
  const params = new URLSearchParams({ start, end });
  return jsonOrThrow(
    await fetch(
      `${BASE}/benchmarks/${encodeURIComponent(benchmarkId)}/series?${params}`,
    ),
    "load benchmark prices",
  );
}

export async function fetchRun(
  config: RunConfig = { population_size: 60, generations: 8 },
): Promise<RunResult> {
  const jobId = await submitRun(config);
  for (let i = 0; i < 600; i++) {
    const status = await getRun(jobId);
    if (status.status === "done" && status.result) return status.result;
    if (status.status === "stopped") {
      if (status.result) return status.result;
      throw new Error("run stopped before a report was completed");
    }
    if (status.status === "failed") throw new Error(status.error ?? "run failed");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("run timed out");
}

// --- extensibility (Phase 7) ---------------------------------------------------
export async function getPrimitives(): Promise<PrimitiveInfo[]> {
  const res = await fetch(`${BASE}/primitives`);
  if (!res.ok) throw new Error(`primitives failed: ${res.status}`);
  return (await res.json()) as PrimitiveInfo[];
}

export async function registerOperator(spec: OperatorSpec): Promise<PrimitiveInfo> {
  const res = await fetch(`${BASE}/operators`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(detail.detail ?? `register failed: ${res.status}`);
  }
  return (await res.json()) as PrimitiveInfo;
}

export async function listFormulas(): Promise<FormulaSpec[]> {
  return jsonOrThrow(await fetch(`${BASE}/formulas`), "list formulas");
}

export async function addFormula(spec: FormulaSpec): Promise<FormulaSpec> {
  return jsonOrThrow(await POST("/formulas", spec), "save formula");
}

export async function updateFormula(
  name: string,
  spec: FormulaSpec,
  strategy: "update" | "upgrade_references" = "update",
): Promise<FormulaSpec> {
  const path = `${BASE}/formulas/${encodeURIComponent(name)}`;
  const res = await fetch(
    strategy === "update" ? path : `${path}?strategy=${encodeURIComponent(strategy)}`,
    {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
    },
  );
  return jsonOrThrow(res, "update formula");
}

export async function getFormula(name: string): Promise<FormulaDetail> {
  return jsonOrThrow(
    await fetch(`${BASE}/formulas/${encodeURIComponent(name)}`),
    "load formula",
  );
}

export async function getFormulaImpact(name: string, spec: FormulaSpec): Promise<FormulaImpact> {
  return jsonOrThrow(
    await POST(`/formulas/${encodeURIComponent(name)}/impact`, spec),
    "check formula references",
  );
}

export async function validateFormula(spec: FormulaSpec): Promise<FormulaValidation> {
  return jsonOrThrow(await POST("/formulas/validate", spec), "validate formula");
}

export async function deleteFormula(name: string): Promise<void> {
  const res = await fetch(`${BASE}/formulas/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`delete formula failed: ${res.status}`);
}

export async function getCategories(): Promise<CategorySettings> {
  return jsonOrThrow(await fetch(`${BASE}/categories`), "load categories");
}

export async function putCategories(update: Partial<CategorySettings>): Promise<CategorySettings> {
  const res = await fetch(`${BASE}/categories`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  return jsonOrThrow(res, "update categories");
}

export async function setPrimitiveCategory(
  primitive: string,
  category: string,
): Promise<CategorySettings> {
  const res = await fetch(`${BASE}/categories/${encodeURIComponent(primitive)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ category }),
  });
  return jsonOrThrow(res, "set category");
}

export async function defineUniverse(
  spec: UniverseSpec,
): Promise<{ name: string; symbols: string[] }> {
  const res = await fetch(`${BASE}/universes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  if (!res.ok) throw new Error(`define universe failed: ${res.status}`);
  return (await res.json()) as { name: string; symbols: string[] };
}

function normalizeUniverseInfo(item: Partial<UniverseInfo> & { name: string }): UniverseInfo {
  return {
    ...item,
    name: item.name,
    memberships: item.memberships ?? [],
    symbols: item.symbols ?? [],
    source: item.source ?? "custom",
  };
}

export async function listUniverses(options: { summary?: boolean } = {}): Promise<UniverseInfo[]> {
  const suffix = options.summary ? "?view=summary" : "";
  const res = await fetch(`${BASE}/universes${suffix}`);
  if (!res.ok) throw new Error(`universes failed: ${res.status}`);
  const items = (await res.json()) as Array<Partial<UniverseInfo> & { name: string }>;
  return items.map(normalizeUniverseInfo);
}

export async function listUniversePresets(): Promise<UniversePreset[]> {
  return jsonOrThrow(await fetch(`${BASE}/universe-presets`), "list universe presets");
}

export async function getUniverse(name: string): Promise<UniverseInfo> {
  const item = await jsonOrThrow<Partial<UniverseInfo> & { name: string }>(
    await fetch(`${BASE}/universes/${encodeURIComponent(name)}`),
    "load universe",
  );
  return normalizeUniverseInfo(item);
}

export async function getUniverseCoverage(
  name: string,
  asOf?: string,
): Promise<UniverseCacheCoverage & { name: string }> {
  const params = new URLSearchParams();
  if (asOf) params.set("as_of", asOf);
  const suffix = params.size ? `?${params}` : "";
  return jsonOrThrow(
    await fetch(`${BASE}/universes/${encodeURIComponent(name)}/coverage${suffix}`),
    "load universe price readiness",
  );
}

export async function updateUniverse(
  name: string,
  spec: UniverseSpec,
): Promise<{ name: string; symbols: string[] }> {
  const res = await fetch(`${BASE}/universes/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  return jsonOrThrow(res, "update universe");
}

export async function deleteUniverse(name: string): Promise<void> {
  const res = await fetch(`${BASE}/universes/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`delete universe failed: ${res.status}`);
}

export async function listUniverseFolders(): Promise<UniverseFolderTree> {
  return jsonOrThrow(await fetch(`${BASE}/universe-folders`), "list universe folders");
}

export async function createUniverseFolder(
  name: string,
  parent: string | null = null,
): Promise<UniverseFolder> {
  return jsonOrThrow(await POST("/universe-folders", { name, parent }), "create folder");
}

/** Rename and/or move a folder; omit a key to keep it, pass `parent: null` for top level. */
export async function updateUniverseFolder(
  id: string,
  changes: { name?: string; parent?: string | null },
): Promise<UniverseFolder> {
  const res = await fetch(`${BASE}/universe-folders/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  return jsonOrThrow(res, "update folder");
}

/** Removes the folder only; its universes and subfolders move up one level. */
export async function deleteUniverseFolder(id: string): Promise<{ removed: string; moved_to: string | null }> {
  const res = await fetch(`${BASE}/universe-folders/${encodeURIComponent(id)}`, { method: "DELETE" });
  return jsonOrThrow(res, "delete folder");
}

export async function moveUniverses(
  universes: string[],
  folder: string | null,
): Promise<{ universes: string[]; folder: string | null }> {
  const res = await fetch(`${BASE}/universe-folders/placements`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ universes, folder }),
  });
  return jsonOrThrow(res, "move universes");
}

export async function searchSymbols(query: string, limit = 15): Promise<SymbolCandidate[]> {
  const params = new URLSearchParams({ query, limit: String(limit) });
  return jsonOrThrow(await fetch(`${BASE}/symbols/search?${params}`), "search symbols");
}

export async function validateSymbol(payload: {
  symbol: string;
  start?: string;
  end?: string;
  force?: boolean;
}): Promise<SymbolValidation> {
  return jsonOrThrow(await POST("/symbols/validate", payload), "validate symbol");
}

export async function getDataCoverage(
  symbols: string[],
  start?: string,
  end?: string,
): Promise<DataCoverage[]> {
  const params = new URLSearchParams({ symbols: symbols.join(",") });
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  return jsonOrThrow(await fetch(`${BASE}/data/coverage?${params}`), "load data coverage");
}

export async function startDataSync(
  req: DataSyncRequest,
): Promise<{ job_id: string; status: string; reused?: boolean }> {
  return jsonOrThrow(await POST("/data/sync", req), "start data sync");
}

export async function getDataSync(jobId: string): Promise<DataSyncJob> {
  return jsonOrThrow(await fetch(`${BASE}/data/sync/${encodeURIComponent(jobId)}`), "load sync job");
}

export async function listDataSyncs(
  options: { activeOnly?: boolean } = {},
): Promise<DataSyncJob[]> {
  const suffix = options.activeOnly ? "?active_only=true" : "";
  return jsonOrThrow(await fetch(`${BASE}/data/sync${suffix}`), "list sync jobs");
}

export async function stopDataSync(jobId: string): Promise<{ stopping: boolean }> {
  return jsonOrThrow(
    await POST(`/data/sync/${encodeURIComponent(jobId)}/stop`, {}),
    "stop sync job",
  );
}

export async function startMembershipSync(
  req: MembershipSyncRequest,
): Promise<{ job_id: string; status: string }> {
  return jsonOrThrow(await POST("/universes/sync-dates", req), "start membership sync");
}

export async function getMembershipSync(jobId: string): Promise<MembershipSyncJob> {
  return jsonOrThrow(
    await fetch(`${BASE}/universes/sync-dates/${encodeURIComponent(jobId)}`),
    "load membership sync job",
  );
}

export async function listWorkspaces(): Promise<WorkspaceSummary[]> {
  const res = await fetch(`${BASE}/workspaces`);
  if (!res.ok) throw new Error(`workspaces failed: ${res.status}`);
  return (await res.json()) as WorkspaceSummary[];
}

export async function saveWorkspace(snapshot: WorkspaceSnapshot): Promise<WorkspaceSnapshot> {
  const res = await fetch(`${BASE}/workspaces`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(snapshot),
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(detail.detail ?? `save workspace failed: ${res.status}`);
  }
  return (await res.json()) as WorkspaceSnapshot;
}

export async function getWorkspace(id: string): Promise<WorkspaceSnapshot> {
  const res = await fetch(`${BASE}/workspaces/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`workspace load failed: ${res.status}`);
  return (await res.json()) as WorkspaceSnapshot;
}

export async function deleteWorkspace(id: string): Promise<void> {
  const res = await fetch(`${BASE}/workspaces/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`workspace delete failed: ${res.status}`);
}

// --- iterative training sessions (A4/A5) ---------------------------------------
export interface SessionHandle {
  session_id: string;
  job_id: string;
}

export async function createSession(req: SessionCreateRequest): Promise<SessionHandle> {
  return jsonOrThrow(await POST("/sessions", req), "create session");
}

export async function continueSession(
  sessionId: string,
  req: SessionContinueRequest,
): Promise<SessionHandle> {
  return jsonOrThrow(await POST(`/sessions/${encodeURIComponent(sessionId)}/continue`, req), "continue session");
}

export async function getSession(sessionId: string): Promise<SessionState> {
  return jsonOrThrow(await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}`), "load session");
}

export async function listSessions(): Promise<SessionSummary[]> {
  return jsonOrThrow(await fetch(`${BASE}/sessions`), "list sessions");
}

export async function listSessionRounds(sessionId: string): Promise<SessionRoundSummary[]> {
  return jsonOrThrow(
    await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/rounds`),
    "list session rounds",
  );
}

export async function getSessionRound(sessionId: string, round: number): Promise<RunResult> {
  return jsonOrThrow(
    await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/rounds/${round}`),
    "load session round",
  );
}

export async function createStrategyComparison(
  sessionId: string,
  round: number,
  strategies: PortfolioStrategySpec[],
  confirmRepeat = false,
): Promise<StrategyComparisonHandle> {
  return jsonOrThrow(
    await POST(
      `/sessions/${encodeURIComponent(sessionId)}/rounds/${round}/strategy-comparisons`,
      {
        strategies,
        ...(confirmRepeat ? { confirm_repeat: true } : {}),
      },
    ),
    "compare validation strategies",
  );
}

export async function listStrategyComparisons(
  sessionId: string,
  round: number,
): Promise<StrategyComparisonSummary[]> {
  return jsonOrThrow(
    await fetch(
      `${BASE}/sessions/${encodeURIComponent(sessionId)}/rounds/${round}/strategy-comparisons`,
    ),
    "list validation strategy comparisons",
  );
}

export async function getStrategyComparison(
  sessionId: string,
  round: number,
  comparisonId: string,
): Promise<StrategyComparisonDetail> {
  return jsonOrThrow(
    await fetch(
      `${BASE}/sessions/${encodeURIComponent(sessionId)}/rounds/${round}/strategy-comparisons/${encodeURIComponent(comparisonId)}`,
    ),
    "load validation strategy comparison",
  );
}

export async function createFinalizationPlan(
  sessionId: string,
  round: number,
  comparisonId: string,
  primaryStrategyId: string,
): Promise<FinalizationPlan> {
  return jsonOrThrow(
    await POST(
      `/sessions/${encodeURIComponent(sessionId)}/rounds/${round}/finalization-plans`,
      {
        comparison_id: comparisonId,
        primary_strategy_id: primaryStrategyId,
      },
    ),
    "pin finalization strategy",
  );
}

export async function listFinalizationPlans(
  sessionId: string,
  round: number,
): Promise<FinalizationPlan[]> {
  return jsonOrThrow(
    await fetch(
      `${BASE}/sessions/${encodeURIComponent(sessionId)}/rounds/${round}/finalization-plans`,
    ),
    "list finalization plans",
  );
}

export async function getFinalizationPlan(
  sessionId: string,
  round: number,
  planId: string,
): Promise<FinalizationPlan> {
  return jsonOrThrow(
    await fetch(
      `${BASE}/sessions/${encodeURIComponent(sessionId)}/rounds/${round}/finalization-plans/${encodeURIComponent(planId)}`,
    ),
    "load finalization plan",
  );
}

export async function finalizeSessionRound(
  sessionId: string,
  round: number,
  confirmRepeat = false,
  strategyPlanId?: string | null,
): Promise<SessionFinalizationHandle> {
  return jsonOrThrow(
    await POST(
      `/sessions/${encodeURIComponent(sessionId)}/rounds/${round}/finalize`,
      {
        confirm_repeat: confirmRepeat,
        ...(strategyPlanId ? { strategy_plan_id: strategyPlanId } : {}),
      },
    ),
    "finalize training round",
  );
}

export async function listSessionFinalizations(
  sessionId: string,
): Promise<SessionFinalizationSummary[]> {
  return jsonOrThrow(
    await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/finalizations`),
    "list holdout finalizations",
  );
}

export async function getSessionFinalization(
  sessionId: string,
  evaluationId: string,
): Promise<SessionFinalizationDetail> {
  return jsonOrThrow(
    await fetch(
      `${BASE}/sessions/${encodeURIComponent(sessionId)}/finalizations/${encodeURIComponent(
        evaluationId,
      )}`,
    ),
    "load holdout finalization",
  );
}

export async function getSessionLineage(sessionId: string, round?: number): Promise<Lineage> {
  const suffix = round === undefined ? "" : `?round=${encodeURIComponent(String(round))}`;
  return jsonOrThrow(
    await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/lineage${suffix}`),
    "load lineage",
  );
}

export async function stopSession(sessionId: string): Promise<{ stopping: boolean }> {
  return jsonOrThrow(await POST(`/sessions/${encodeURIComponent(sessionId)}/stop`, {}), "stop session");
}

export async function getTrainingCapabilities(): Promise<TrainingCapabilities> {
  return jsonOrThrow(await fetch(`${BASE}/training/capabilities`), "load training capabilities");
}

// --- saved factors (A3) --------------------------------------------------------
export async function listFactors(): Promise<SavedFactor[]> {
  return jsonOrThrow(await fetch(`${BASE}/factors`), "list factors");
}

export async function saveFactor(payload: {
  name: string;
  tree: unknown;
  metrics?: Record<string, number>;
  provenance?: Record<string, unknown>;
  notes?: string;
}): Promise<SavedFactor> {
  return jsonOrThrow(await POST("/factors", payload), "save factor");
}

export async function deleteFactor(id: string): Promise<void> {
  const res = await fetch(`${BASE}/factors/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`delete factor failed: ${res.status}`);
}

export async function renameFactor(id: string, name: string): Promise<SavedFactor> {
  const res = await fetch(`${BASE}/factors/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return jsonOrThrow(res, "rename factor");
}

// --- formula tests and canonical results --------------------------------------
export async function startFormulaTest(payload: FormulaTestRequest): Promise<FormulaTestJob> {
  return jsonOrThrow(await POST("/formula-tests", payload), "start formula backtest");
}

export async function listFormulaTests(): Promise<FormulaTestJob[]> {
  return jsonOrThrow(await fetch(`${BASE}/formula-tests`), "list formula backtests");
}

export async function getFormulaTest(jobId: string): Promise<FormulaTestJob> {
  return jsonOrThrow(
    await fetch(`${BASE}/formula-tests/${encodeURIComponent(jobId)}`),
    "load formula backtest",
  );
}

export async function stopFormulaTest(jobId: string): Promise<{ stopping: boolean }> {
  return jsonOrThrow(
    await POST(`/formula-tests/${encodeURIComponent(jobId)}/stop`, {}),
    "stop formula backtest",
  );
}

export async function clearFormulaTest(jobId: string): Promise<void> {
  const res = await fetch(`${BASE}/formula-tests/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`clear formula backtest failed: ${res.status}`);
}

export async function keepFormulaTest(
  jobId: string,
  payload: { name: string; notes?: string },
): Promise<FormulaResult> {
  return jsonOrThrow(
    await POST(`/formula-tests/${encodeURIComponent(jobId)}/keep`, payload),
    "keep formula result",
  );
}

export async function listFormulaResults(): Promise<FormulaResult[]> {
  return jsonOrThrow(await fetch(`${BASE}/formula-results`), "list formula results");
}

export async function getFormulaResult(id: string): Promise<FormulaResult> {
  return jsonOrThrow(
    await fetch(`${BASE}/formula-results/${encodeURIComponent(id)}`),
    "load formula result",
  );
}

export async function updateFormulaResult(
  id: string,
  patch: { name?: string; notes?: string },
): Promise<FormulaResult> {
  const res = await fetch(`${BASE}/formula-results/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return jsonOrThrow(res, "update formula result");
}

export async function deleteFormulaResult(id: string): Promise<void> {
  const res = await fetch(`${BASE}/formula-results/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`delete formula result failed: ${res.status}`);
}

// --- settings ------------------------------------------------------------------
export async function getSettings(): Promise<Settings> {
  return jsonOrThrow(await fetch(`${BASE}/settings`), "load settings");
}

export async function putSettings(update: SettingsUpdate): Promise<Settings> {
  const res = await fetch(`${BASE}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  return jsonOrThrow(res, "update settings");
}

// --- local data usage + cleanup ------------------------------------------------
export async function getDataUsage(): Promise<DataUsageRow[]> {
  return jsonOrThrow(await fetch(`${BASE}/data/usage`), "load data usage");
}

export async function clearData(category: string): Promise<DataUsageRow> {
  return jsonOrThrow(await POST("/data/clear", { category }), "clear data");
}

// --- P11: the agent ------------------------------------------------------------
/** The complete tool catalog. Shown before a message so the user sees what they authorize. */
export async function getAgentTools(): Promise<AgentToolCatalog> {
  return jsonOrThrow(await fetch(`${BASE}/agent/tools`), "load agent tools");
}

export async function getConversation(sessionId: string): Promise<Conversation> {
  return jsonOrThrow(
    await fetch(`${BASE}/agent/conversations/${sessionId}`),
    "load conversation",
  );
}

export async function clearConversation(sessionId: string): Promise<{ cleared: boolean }> {
  const res = await fetch(`${BASE}/agent/conversations/${sessionId}`, { method: "DELETE" });
  return jsonOrThrow(res, "clear conversation");
}

export async function sendAgentMessage(
  sessionId: string,
  req: AgentMessageRequest,
): Promise<{ job_id: string }> {
  return jsonOrThrow(
    await POST(`/agent/conversations/${sessionId}/messages`, req),
    "send message",
  );
}

export async function getAgentJob(jobId: string): Promise<AgentJob> {
  return jsonOrThrow(await fetch(`${BASE}/agent/jobs/${jobId}`), "load agent job");
}

export async function stopAgentTurn(jobId: string): Promise<{ stopping: boolean }> {
  return jsonOrThrow(await POST(`/agent/jobs/${jobId}/stop`, {}), "stop agent");
}

/**
 * The approval gate for a factor: runs the session's real validation pass, folds the agent's
 * trials into the session, and writes a new round.
 */
export async function promoteAgentProposal(
  sessionId: string,
  proposalId: string,
): Promise<{ job_id: string }> {
  return jsonOrThrow(
    await POST(`/agent/conversations/${sessionId}/proposals/${proposalId}/promote`, {}),
    "promote proposal",
  );
}

export async function applyAgentConfig(
  sessionId: string,
  proposalId: string,
): Promise<Record<string, unknown>> {
  return jsonOrThrow(
    await POST(`/agent/conversations/${sessionId}/proposals/${proposalId}/apply-config`, {}),
    "apply config patch",
  );
}

export async function deleteSession(sessionId: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/sessions/${sessionId}`, { method: "DELETE" });
  return jsonOrThrow(res, "delete session");
}

// --- shutdown (single-process launcher Quit) -----------------------------------
export async function shutdown(): Promise<{ shutting_down: boolean }> {
  return jsonOrThrow(await POST("/shutdown", {}), "shutdown");
}

export async function startRoundOverlap(
  sessionId: string,
  roundIndex: number,
  body: { include_saved_results?: boolean; top_k?: number } = {},
): Promise<{ job_id: string; status: string; reused: boolean }> {
  return jsonOrThrow(
    await POST(
      `/sessions/${encodeURIComponent(sessionId)}/rounds/${roundIndex}/overlap`,
      body,
    ),
    "start overlap check",
  );
}

/** The last stored overlap check for a round, or null when none has been run. */
export async function getRoundOverlap(
  sessionId: string,
  roundIndex: number,
): Promise<FactorOverlapReport | null> {
  const res = await fetch(
    `${BASE}/sessions/${encodeURIComponent(sessionId)}/rounds/${roundIndex}/overlap`,
  );
  if (res.status === 404) return null;
  return jsonOrThrow(res, "load overlap check");
}
