import { useCallback, useEffect, useState } from "react";
import {
  getSessionLineage,
  getUniverse,
  getWorkspace,
  listWorkspaces,
  saveFactor,
  saveWorkspace,
  shutdown,
  stopSession,
} from "../api/client";
import { loadRun } from "../api/dataSource";
import {
  parseFactor,
  type FormulaDraft,
  type LineageNode,
  type OperatorComposerDraft,
  type RunResult,
  type SyncProgressSnapshot,
  type UniverseDraft,
  type WorkspaceSnapshot,
} from "../api/types";
import { Dashboard } from "../dashboard/Dashboard";
import { ExtendPanel, type ExtendPage } from "../extend/ExtendPanel";
import { rowsFromUniverse } from "../extend/toUniversePayload";
import { FactorDetail } from "../factor/FactorDetail";
import { FactorTree } from "../factor/FactorTree";
import type { TreeNodeData } from "../factor/treeToFlow";
import { Genealogy } from "../genealogy/Genealogy";
import { LineageDetail } from "../genealogy/LineageDetail";
import { LibraryPanel } from "../library/LibraryPanel";
import { TrainPanel } from "../train/TrainPanel";
import { AppShell, type Tab } from "./AppShell";
import { ErrorBoundary } from "./ErrorBoundary";
import { getAppMode } from "./mode";
import {
  makeWorkspaceSnapshot,
  readLocalWorkspace,
  writeLocalWorkspace,
} from "./workspace";

function tabFromWorkspace(snapshot: WorkspaceSnapshot | null, mode: string): Tab {
  if (snapshot?.ui.selectedTab) return snapshot.ui.selectedTab;
  // App mode with nothing loaded starts at the run launcher; demo opens on metrics.
  return mode === "app" && !snapshot?.run ? "train" : "dashboard";
}

function applyNode(node?: { name: string; value?: number } | null): TreeNodeData | null {
  return node ? { name: node.name, value: node.value } : null;
}

export function App() {
  const mode = getAppMode();
  const [initialWorkspace] = useState(() => readLocalWorkspace());
  const [run, setRun] = useState<RunResult | null>(initialWorkspace?.run ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<Tab>(tabFromWorkspace(initialWorkspace, mode));
  const [selectedNode, setSelectedNode] = useState<TreeNodeData | null>(
    applyNode(initialWorkspace?.ui.selectedFactorNode),
  );
  const [selectedLineage, setSelectedLineage] = useState<number | null>(
    initialWorkspace?.ui.selectedLineage ?? null,
  );
  const [universeDraft, setUniverseDraft] = useState<UniverseDraft | undefined>(
    initialWorkspace?.universeDraft,
  );
  const [operatorDraft, setOperatorDraft] = useState<OperatorComposerDraft | undefined>(
    initialWorkspace?.operatorDraft,
  );
  const [formulaDraft, setFormulaDraft] = useState<FormulaDraft | undefined>(
    initialWorkspace?.formulaDraft,
  );
  const [seedIds, setSeedIds] = useState<string[]>([]);
  const [searchRunning, setSearchRunning] = useState(false);
  const [runningSessionId, setRunningSessionId] = useState<string | null>(null);
  const [bestFactorSaved, setBestFactorSaved] = useState(false);
  const [quitOpen, setQuitOpen] = useState(false);
  const [shutDown, setShutDown] = useState(false);
  const [extendPage, setExtendPage] = useState<ExtendPage>("universe");
  const [dataPullProgress, setDataPullProgress] = useState<SyncProgressSnapshot | null>(null);
  const [status, setStatus] = useState<string | null>(
    initialWorkspace?.run ? "Loaded local workspace" : null,
  );

  const currentSnapshot = useCallback(
    () =>
      makeWorkspaceSnapshot({
        run,
        universeDraft,
        formulaDraft,
        operatorDraft,
        ui: {
          selectedTab: tab,
          selectedFactorNode: selectedNode
            ? { name: selectedNode.name, value: selectedNode.value }
            : null,
          selectedLineage,
          sessionId: run?.session_id ?? null,
        },
      }),
    [formulaDraft, operatorDraft, run, selectedLineage, selectedNode, tab, universeDraft],
  );

  const applyWorkspace = useCallback((snapshot: WorkspaceSnapshot) => {
    setRun(snapshot.run);
    setTab(snapshot.ui.selectedTab ?? "dashboard");
    setSelectedNode(applyNode(snapshot.ui.selectedFactorNode));
    setSelectedLineage(snapshot.ui.selectedLineage ?? null);
    setUniverseDraft(snapshot.universeDraft);
    setFormulaDraft(snapshot.formulaDraft);
    setOperatorDraft(snapshot.operatorDraft);
    setStatus(`Loaded ${snapshot.name}`);
  }, []);

  // Demo mode auto-loads the static snapshot; app mode waits for the user to launch a run.
  const refreshDemo = useCallback(() => {
    setLoading(true);
    setError(null);
    setStatus("Loading static demo...");
    loadRun()
      .then((result) => {
        setRun(result);
        setStatus("Static demo loaded");
      })
      .catch((e) => {
        setError(String(e));
        setStatus("Load failed");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (initialWorkspace?.run) return;
    if (mode === "demo") refreshDemo();
  }, [initialWorkspace?.run, mode, refreshDemo]);

  useEffect(() => {
    writeLocalWorkspace(currentSnapshot());
  }, [currentSnapshot]);

  async function onRunComplete(result: RunResult) {
    // GET /sessions/{id} strips the (large) lineage from the result; fetch it so the Genealogy
    // view has data. Demo runs already carry lineage and have no session_id, so they skip this.
    let full = result;
    if (result.session_id && !result.lineage?.nodes?.length) {
      try {
        full = { ...result, lineage: await getSessionLineage(result.session_id) };
      } catch {
        // keep the stripped result; the genealogy view will show an empty state, not crash
      }
    }
    setRun(full);
    setBestFactorSaved(false); // a fresh best factor is not yet in the library
    setTab("dashboard");
    setStatus("Run completed - showing out-of-sample metrics");
  }

  function onRunningChange(running: boolean, sessionId: string | null) {
    setSearchRunning(running);
    setRunningSessionId(sessionId);
  }

  async function performShutdown() {
    try {
      await shutdown();
    } catch {
      // the server may drop the connection as it exits; that's expected
    }
    setQuitOpen(false);
    setShutDown(true);
  }

  async function stopRunningSearch() {
    if (runningSessionId) {
      try {
        await stopSession(runningSessionId);
      } catch (e) {
        setStatus(String(e));
      }
    }
  }

  function onRefreshRun() {
    if (mode === "app") {
      setTab("train");
    } else {
      refreshDemo();
    }
  }

  function startSeededSession(ids: string[]) {
    setSeedIds(ids);
    setTab("train");
    setStatus(`Seeding a new session from ${ids.length} factor(s)`);
  }

  async function openUniverseEditor(universeName: string) {
    try {
      const universe = await getUniverse(universeName);
      setUniverseDraft({
        name: universe.name,
        rows: rowsFromUniverse(universe),
        selectedUniverse: universe.source === "custom" ? universe.name : "",
        expectedStart: universeDraft?.expectedStart,
      });
    } catch (e) {
      setStatus(String(e));
    }
    setExtendPage("universe");
    setTab("extend");
  }

  async function saveLineageNode(node: LineageNode) {
    try {
      await saveFactor({
        name: `gen${node.generation}-#${node.id}`,
        tree: node.tree,
        metrics: typeof node.fitness === "number" ? { fitness: node.fitness } : {},
        provenance: {
          session_id: run?.session_id,
          generation: node.generation,
          cumulative_trials: run?.cumulative_trials,
          test_reads: run?.test_reads,
        },
      });
      setStatus(`Saved gen${node.generation}-#${node.id} to library`);
    } catch (e) {
      setStatus(String(e));
    }
  }

  async function saveBestFactor() {
    if (!run) return;
    try {
      await saveFactor({
        name: "best factor",
        tree: parseFactor(run.best_factor),
        metrics: { oos_ic: run.report.oos_ic, deflated_sharpe: run.report.deflated_sharpe },
        provenance: {
          session_id: run.session_id,
          cumulative_trials: run.cumulative_trials,
          test_reads: run.test_reads,
        },
      });
      setBestFactorSaved(true);
      setStatus("Saved best factor to library");
    } catch (e) {
      setStatus(String(e));
    }
  }

  function saveLocal() {
    writeLocalWorkspace(currentSnapshot());
    setStatus("Saved local workspace");
  }

  function loadLocal() {
    const snapshot = readLocalWorkspace();
    if (!snapshot) {
      setStatus("No local workspace found");
      return;
    }
    applyWorkspace(snapshot);
  }

  async function saveBackend() {
    if (mode !== "app") return;
    try {
      const saved = await saveWorkspace({
        ...currentSnapshot(),
        id: undefined,
        name: "Backend Workspace",
      });
      setStatus(`Saved backend workspace ${saved.id}`);
    } catch (e) {
      setStatus(String(e));
    }
  }

  async function loadBackend() {
    if (mode !== "app") return;
    try {
      const [latest] = await listWorkspaces();
      if (!latest) {
        setStatus("No backend workspaces found");
        return;
      }
      applyWorkspace(await getWorkspace(latest.id));
    } catch (e) {
      setStatus(String(e));
    }
  }

  const factor = run ? parseFactor(run.best_factor) : null;

  if (shutDown) {
    return (
      <div className="goodbye" data-testid="goodbye">
        <h1>AlphaLineage has shut down.</h1>
        <p>The backend and UI server have stopped. You can close this tab.</p>
      </div>
    );
  }

  const quitWarnings: string[] = [];
  if (searchRunning) quitWarnings.push("A search is still running.");
  if (run && !bestFactorSaved) quitWarnings.push("The best factor isn't saved to your library.");

  return (
    <AppShell
      mode={mode}
      tab={tab}
      status={status}
      progress={dataPullProgress}
      onTabChange={setTab}
      onRefreshRun={onRefreshRun}
      onSaveLocal={saveLocal}
      onLoadLocal={loadLocal}
      onSaveBackend={saveBackend}
      onLoadBackend={loadBackend}
      onQuit={() => setQuitOpen(true)}
      onSelectExtendPage={(page) => {
        setExtendPage(page);
        setTab("extend");
      }}
    >
      <section className="app-page">
        <header className="view-head">
          <div className="view-tag">
            <span className="view-tag__mark" aria-hidden="true" />
            <span>AlphaLineage</span>
          </div>
          <h1>Honest factor evolution</h1>
          <p>Trace each generated alpha from metrics to tree structure to genetic lineage.</p>
          <div className="view-rule" aria-hidden="true" />
        </header>

        {error && <p className="error surface-message">{error}</p>}
        {loading && !run && <p className="surface-message">Loading...</p>}

        <ErrorBoundary key={tab}>
        {tab === "train" && (
          <section className="view-card" data-view="train">
            <div className="view-rail">
              <span>00</span>
              <span />
            </div>
            <div className="view-body">
              <TrainPanel
                seedIds={seedIds}
                restoreSessionId={initialWorkspace?.ui.sessionId ?? null}
                onComplete={onRunComplete}
                onRunningChange={onRunningChange}
                onOpenDashboard={() => setTab("dashboard")}
                onOpenUniverseEditor={openUniverseEditor}
                onOpenFormulaEditor={() => {
                  setExtendPage("formula");
                  setTab("extend");
                }}
              />
            </div>
          </section>
        )}

        {run && tab === "dashboard" && (
          <section className="view-card" data-view="dashboard">
            <div className="view-rail">
              <span>01</span>
              <span />
            </div>
            <div className="view-body">
              <Dashboard report={run.report} history={run.history} extra={run} />
            </div>
          </section>
        )}

        {run && factor && tab === "factor" && (
          <section className="view-card" data-view="factor">
            <div className="view-rail">
              <span>02</span>
              <span />
            </div>
            <div className="view-body split">
              <FactorTree factor={factor} onSelect={setSelectedNode} />
              <div>
                <FactorDetail node={selectedNode} />
                {mode === "app" && (
                  <button
                    type="button"
                    className="ghost"
                    data-testid="save-best-factor"
                    onClick={saveBestFactor}
                  >
                    Save best factor to library
                  </button>
                )}
              </div>
            </div>
          </section>
        )}

        {run && tab === "genealogy" && (
          <section className="view-card" data-view="genealogy">
            <div className="view-rail">
              <span>03</span>
              <span />
            </div>
            <div className="view-body split">
              <Genealogy lineage={run.lineage} onSelect={setSelectedLineage} />
              <LineageDetail
                lineage={run.lineage}
                selectedId={selectedLineage}
                onSelect={setSelectedLineage}
                onSave={mode === "app" ? saveLineageNode : undefined}
              />
            </div>
          </section>
        )}

        {tab === "library" && (
          <section className="view-card" data-view="library">
            <div className="view-rail">
              <span>05</span>
              <span />
            </div>
            <div className="view-body">
              <LibraryPanel onSeed={startSeededSession} />
            </div>
          </section>
        )}

        {tab === "extend" && (
          <section className="view-card" data-view="extend">
            <div className="view-rail">
              <span>04</span>
              <span />
            </div>
            <div className="view-body">
              <ExtendPanel
                page={extendPage}
                universeDraft={universeDraft}
                onUniverseDraftChange={setUniverseDraft}
                operatorDraft={operatorDraft}
                onOperatorDraftChange={setOperatorDraft}
                canSubmit={mode === "app"}
                onDataPullProgressChange={setDataPullProgress}
              />
            </div>
          </section>
        )}
        </ErrorBoundary>

        {quitOpen && (
          <div className="quit-backdrop" role="dialog" aria-modal="true" data-testid="quit-dialog">
            <div className="quit-dialog">
              <h3>Quit AlphaLineage?</h3>
              {quitWarnings.length > 0 ? (
                <ul className="quit-warnings">
                  {quitWarnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              ) : (
                <p>This will shut down the backend and the UI server.</p>
              )}
              <div className="quit-actions">
                {run && !bestFactorSaved && (
                  <button
                    type="button"
                    className="ghost"
                    data-testid="quit-save-factor"
                    onClick={saveBestFactor}
                  >
                    Save best factor
                  </button>
                )}
                {searchRunning && (
                  <button
                    type="button"
                    className="ghost"
                    data-testid="quit-stop-search"
                    onClick={stopRunningSearch}
                  >
                    Stop search
                  </button>
                )}
                <button
                  type="button"
                  className="primary-action"
                  data-testid="quit-confirm"
                  onClick={performShutdown}
                >
                  Quit now
                </button>
                <button type="button" className="ghost" onClick={() => setQuitOpen(false)}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </section>
    </AppShell>
  );
}
