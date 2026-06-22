// Typed client for the `app` build: submit a GP run and poll it to completion.

import type {
  CategorySettings,
  DataCoverage,
  DataSyncJob,
  DataSyncRequest,
  DataUsageRow,
  FormulaSpec,
  FormulaValidation,
  Lineage,
  MembershipSyncJob,
  MembershipSyncRequest,
  OperatorSpec,
  RunResult,
  PrimitiveInfo,
  SavedFactor,
  SessionContinueRequest,
  SessionCreateRequest,
  SessionState,
  SessionSummary,
  Settings,
  SettingsUpdate,
  SymbolCandidate,
  SymbolValidation,
  UniverseInfo,
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

export async function fetchRun(
  config: RunConfig = { population_size: 60, generations: 8 },
): Promise<RunResult> {
  const jobId = await submitRun(config);
  for (let i = 0; i < 600; i++) {
    const status = await getRun(jobId);
    if (status.status === "done" && status.result) return status.result;
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

export async function updateFormula(name: string, spec: FormulaSpec): Promise<FormulaSpec> {
  const res = await fetch(`${BASE}/formulas/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  return jsonOrThrow(res, "update formula");
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

export async function listUniverses(): Promise<UniverseInfo[]> {
  const res = await fetch(`${BASE}/universes`);
  if (!res.ok) throw new Error(`universes failed: ${res.status}`);
  return (await res.json()) as UniverseInfo[];
}

export async function getUniverse(name: string): Promise<UniverseInfo> {
  return jsonOrThrow(await fetch(`${BASE}/universes/${encodeURIComponent(name)}`), "load universe");
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

export async function startDataSync(req: DataSyncRequest): Promise<{ job_id: string; status: string }> {
  return jsonOrThrow(await POST("/data/sync", req), "start data sync");
}

export async function getDataSync(jobId: string): Promise<DataSyncJob> {
  return jsonOrThrow(await fetch(`${BASE}/data/sync/${encodeURIComponent(jobId)}`), "load sync job");
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

export async function getSessionLineage(sessionId: string): Promise<Lineage> {
  return jsonOrThrow(await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/lineage`), "load lineage");
}

export async function stopSession(sessionId: string): Promise<{ stopping: boolean }> {
  return jsonOrThrow(await POST(`/sessions/${encodeURIComponent(sessionId)}/stop`, {}), "stop session");
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

// --- shutdown (single-process launcher Quit) -----------------------------------
export async function shutdown(): Promise<{ shutting_down: boolean }> {
  return jsonOrThrow(await POST("/shutdown", {}), "shutdown");
}
