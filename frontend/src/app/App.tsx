import { useCallback, useEffect, useRef, useState } from "react";
import {
  finalizeSessionRound,
  getSessionFinalization,
  getSessionLineage,
  getSessionRound,
  getUniverse,
  getWorkspace,
  listSessionFinalizations,
  listSessionRounds,
  listWorkspaces,
  saveFactor,
  saveWorkspace,
  shutdown,
} from "../api/client";
import { loadRun } from "../api/dataSource";
import {
  parseFactor,
  type FormulaDraft,
  type LineageNode,
  type OperatorComposerDraft,
  type RunResult,
  type SessionFinalizationSummary,
  type SessionRoundSummary,
  type SyncProgressSnapshot,
  type UniverseDraft,
  type WorkspaceSnapshot,
} from "../api/types";
import { Dashboard } from "../dashboard/Dashboard";
import { AgentPage } from "../agent/AgentPage";
import { ExtendPanel, type ExtendPage } from "../extend/ExtendPanel";
import { rowsFromUniverse } from "../extend/toUniversePayload";
import { BestFormulaResultPage } from "../factor/BestFormulaResultPage";
import type { TreeNodeData } from "../factor/treeToFlow";
import { Genealogy } from "../genealogy/Genealogy";
import { LibraryPanel } from "../library/LibraryPanel";
import { TrainPanel } from "../train/TrainPanel";
import { useSession } from "../train/useSession";
import { AppShell, isTab, type Tab } from "./AppShell";
import { ErrorBoundary } from "./ErrorBoundary";
import { EvaluationNavigator } from "./EvaluationNavigator";
import { getAppMode } from "./mode";
import { PageHeader } from "./PageHeader";
import { RoundNavigator } from "./RoundNavigator";
import {
  makeWorkspaceSnapshot,
  migrateFormulaDrafts,
  readLocalWorkspace,
  writeLocalWorkspace,
} from "./workspace";

const PAGE_COPY: Record<Exclude<Tab, "extend">, { title: string; description: string }> = {
  train: {
    title: "Train",
    description: "Configure a reproducible search, continue the latest checkpoint, and preserve every validation round.",
  },
  dashboard: {
    title: "Metrics",
    description: "Review validation robustness first, then explicitly evaluate a selected round on the locked holdout.",
  },
  factor: {
    title: "Best Formula Result",
    description: "Inspect the immutable formula selected by training and validation evidence for this round.",
  },
  genealogy: {
    title: "Genealogy",
    description: "Trace the selected formula through generations, operations, parents, and retained champions.",
  },
  library: {
    title: "Formula Results",
    description: "Review saved training and backtest evidence, or seed a new search from selected results.",
  },
  agent: {
    title: "Agent",
    description: "Ask about a discovered formula, or put a model to work improving it. It proposes; you decide.",
  },
};

function tabFromWorkspace(snapshot: WorkspaceSnapshot | null, mode: string): Tab {
  // Checked rather than trusted: a workspace saved by an older build can name a tab that has
  // since been renamed or removed, and a bad value here reaches PAGE_COPY below.
  if (isTab(snapshot?.ui.selectedTab)) return snapshot.ui.selectedTab;
  // App mode with nothing loaded starts at the run launcher; demo opens on metrics.
  return mode === "app" && !snapshot?.run ? "train" : "dashboard";
}

function applyNode(node?: { name: string; value?: number } | null): TreeNodeData | null {
  return node ? { name: node.name, value: node.value } : null;
}

function defaultUniverseHistoryStart(universe: Awaited<ReturnType<typeof getUniverse>>): string {
  if (universe.mode === "static_snapshot") return "2020-01-01";
  return universe.memberships
    .map((membership) => membership.entry)
    .filter(Boolean)
    .sort()[0] ?? "2020-01-01";
}

function isSelectableRound(round: SessionRoundSummary): boolean {
  return (
    round.status === "done" ||
    round.status === "completed" ||
    round.report_available ||
    round.finalization_available === true
  );
}

export function App() {
  const mode = getAppMode();
  const [initialWorkspace] = useState(() => readLocalWorkspace());
  const [run, setRun] = useState<RunResult | null>(initialWorkspace?.run ?? null);
  const [rounds, setRounds] = useState<SessionRoundSummary[]>([]);
  const [evaluations, setEvaluations] = useState<SessionFinalizationSummary[]>([]);
  const [selectedEvaluation, setSelectedEvaluation] = useState<string | null>(null);
  const [selectedRound, setSelectedRound] = useState<number | null>(
    initialWorkspace?.ui.selectedRound ??
      initialWorkspace?.run?.round_index ??
      initialWorkspace?.run?.segment ??
      null,
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
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
    () => migrateFormulaDrafts(initialWorkspace?.formulaDraft).active,
  );
  const [recoveredFormulaDrafts, setRecoveredFormulaDrafts] = useState(
    () => migrateFormulaDrafts(initialWorkspace?.formulaDraft).recoveries,
  );
  const [seedIds, setSeedIds] = useState<string[]>([]);
  const [bestFactorSaved, setBestFactorSaved] = useState(false);
  const [quitOpen, setQuitOpen] = useState(false);
  const [shutDown, setShutDown] = useState(false);
  const [extendPage, setExtendPage] = useState<ExtendPage>("universe");
  const [dataPullProgress, setDataPullProgress] = useState<SyncProgressSnapshot | null>(null);
  const [status, setStatus] = useState<string | null>(
    initialWorkspace?.run ? "Loaded local workspace" : null,
  );
  const handledFinalizationRef = useRef<string | null>(null);
  const sessionController = useSession(
    onRunComplete,
    initialWorkspace?.ui.sessionId ?? initialWorkspace?.run?.session_id ?? null,
  );
  const searchRunning = sessionController.phase === "running";
  const runningSessionId = sessionController.sessionId;

  const currentSnapshot = useCallback(
    () =>
      makeWorkspaceSnapshot({
        run,
        universeDraft,
        formulaDraft,
        recoveredFormulaDrafts,
        operatorDraft,
        ui: {
          selectedTab: tab,
          selectedFactorNode: selectedNode
            ? { name: selectedNode.name, value: selectedNode.value }
            : null,
          selectedLineage,
          sessionId: sessionController.sessionId ?? run?.session_id ?? null,
          selectedRound,
        },
      }),
    [
      formulaDraft,
      operatorDraft,
      recoveredFormulaDrafts,
      run,
      selectedLineage,
      selectedNode,
      selectedRound,
      sessionController.sessionId,
      tab,
      universeDraft,
    ],
  );

  const applyWorkspace = useCallback((snapshot: WorkspaceSnapshot) => {
    setRun(snapshot.run);
    setTab(isTab(snapshot.ui.selectedTab) ? snapshot.ui.selectedTab : "dashboard");
    setSelectedNode(applyNode(snapshot.ui.selectedFactorNode));
    setSelectedLineage(snapshot.ui.selectedLineage ?? null);
    setUniverseDraft(snapshot.universeDraft);
    const migratedDrafts = migrateFormulaDrafts(snapshot.formulaDraft);
    setFormulaDraft(migratedDrafts.active);
    setRecoveredFormulaDrafts(migratedDrafts.recoveries);
    setOperatorDraft(snapshot.operatorDraft);
    setSelectedRound(
      snapshot.ui.selectedRound ?? snapshot.run?.round_index ?? snapshot.run?.segment ?? null,
    );
    const restoredSessionId = snapshot.ui.sessionId ?? snapshot.run?.session_id ?? null;
    if (restoredSessionId) sessionController.attach(restoredSessionId);
    else sessionController.reset();
    setStatus(`Loaded ${snapshot.name}`);
  }, [sessionController]);

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
    const sessionId = sessionController.sessionId;
    if (!sessionId) {
      setRounds([]);
      return;
    }
    let active = true;
    listSessionRounds(sessionId)
      .then((items) => {
        if (!active) return;
        setRounds(items);
        const available = items.filter(isSelectableRound);
        const latest = available[available.length - 1];
        setSelectedRound((current) =>
          current !== null && available.some((item) => item.index === current)
            ? current
            : latest?.index ?? null,
        );
      })
      .catch(() => {
        if (active) setRounds([]);
      });
    return () => {
      active = false;
    };
  }, [sessionController.sessionId, sessionController.state?.segments.length]);

  useEffect(() => {
    writeLocalWorkspace(currentSnapshot());
  }, [currentSnapshot]);

  useEffect(() => {
    const job =
      sessionController.state?.finalization_job ??
      sessionController.state?.last_finalization_job;
    if (!job) return;
    if (job.status === "failed" || job.status === "stopped") {
      setFinalizing(false);
      return;
    }
    if (job.status !== "done") return;
    const metadata = job.metadata ?? {};
    const evaluationId =
      (metadata.evaluation_id as string | undefined) ?? job.evaluation_id;
    const roundIndex =
      (metadata.round_index as number | undefined) ?? job.round_index;
    const sessionId = sessionController.sessionId;
    if (!sessionId || !evaluationId || roundIndex === undefined) return;
    const handledKey = `${sessionId}:${evaluationId}`;
    if (handledFinalizationRef.current === handledKey) return;
    handledFinalizationRef.current = handledKey;

    setFinalizing(true);
    void loadRoundWithFinalization(sessionId, roundIndex, evaluationId)
      .then(async (result) => {
        setRun(result);
        setSelectedRound(roundIndex);
        setRounds(await listSessionRounds(sessionId));
        setStatus(
          result.evidence_status === "locked_first_read"
            ? "Showing the first locked-holdout evaluation"
            : "Showing exploratory repeated-holdout evidence",
        );
      })
      .catch((reason) => {
        setStatus(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => setFinalizing(false));
  }, [
    sessionController.sessionId,
    sessionController.state?.finalization_job,
    sessionController.state?.last_finalization_job,
  ]);

  async function resultWithLineage(result: RunResult, round?: number): Promise<RunResult> {
    if (!result.session_id || (round === undefined && result.lineage?.nodes?.length)) {
      return result;
    }
    try {
      return {
        ...result,
        lineage:
          round === undefined
            ? await getSessionLineage(result.session_id)
            : await getSessionLineage(result.session_id, round),
      };
    } catch {
      return result;
    }
  }

  async function loadRoundWithFinalization(
    sessionId: string,
    roundIndex: number,
    preferredEvaluationId?: string | null,
  ): Promise<RunResult> {
    const validationRound = await getSessionRound(sessionId, roundIndex);
    let merged = validationRound;
    try {
      const finalizations = await listSessionFinalizations(sessionId);
      const matchingFinalizations = finalizations.filter(
        (item) => item.round_index === roundIndex && item.report_available,
      );
      setEvaluations(matchingFinalizations);
      const selected =
        matchingFinalizations.find(
          (item) => item.evaluation_id === preferredEvaluationId,
        ) ??
        matchingFinalizations.find((item) => item.evidence_status === "locked_first_read") ??
        matchingFinalizations[0];
      setSelectedEvaluation(selected?.evaluation_id ?? null);
      if (selected) {
        const detail = await getSessionFinalization(sessionId, selected.evaluation_id);
        merged = {
          ...validationRound,
          best_factor: detail.best_factor ?? validationRound.best_factor,
          report: detail.report,
          oos_backtest: detail.oos_backtest ?? detail.report.oos_backtest ?? null,
          context: detail.context ?? validationRound.context,
          selection: detail.selection ?? validationRound.selection,
          evidence_status: detail.evidence_status,
          validation_only: false,
          finalization: detail,
          test_reads: detail.test_reads,
          session_holdout_reads: detail.session_holdout_reads,
          holdout_fingerprint: detail.holdout_fingerprint,
          same_holdout_read_index: detail.same_holdout_read_index,
          test_read_index: detail.same_holdout_read_index,
          inherited_evidence_sources: detail.inherited_evidence_sources,
          strategy_results: detail.strategy_results,
          primary_strategy_id: detail.primary_strategy_id,
          strategy_plan_id: detail.strategy_plan_id,
          comparison_id: detail.comparison_id,
        };
      }
    } catch {
      // A validation round remains useful if an older backend has no finalization index yet.
      setEvaluations([]);
      setSelectedEvaluation(null);
    }
    return resultWithLineage(merged, roundIndex);
  }

  async function onRunComplete(result: RunResult) {
    const roundIndex = result.round_index ?? result.segment;
    const full =
      result.session_id && roundIndex !== undefined
        ? await loadRoundWithFinalization(result.session_id, roundIndex)
        : await resultWithLineage(result, roundIndex);
    setRun(full);
    if (result.session_id) {
      try {
        const items = await listSessionRounds(result.session_id);
        setRounds(items);
        const available = items.filter(isSelectableRound);
        const latest = available[available.length - 1];
        setSelectedRound(latest?.index ?? roundIndex ?? null);
      } catch {
        setSelectedRound(roundIndex ?? null);
      }
    }
    setBestFactorSaved(false); // a fresh best formula result is not yet in the library
    setTab("dashboard");
    setStatus("Validation round completed - the locked holdout remains unopened");
  }

  async function selectRound(roundIndex: number) {
    const sessionId = sessionController.sessionId ?? run?.session_id;
    if (!sessionId || roundIndex === selectedRound) return;
    setLoading(true);
    try {
      setRun(await loadRoundWithFinalization(sessionId, roundIndex));
      setSelectedRound(roundIndex);
      setSelectedNode(null);
      setSelectedLineage(null);
      setBestFactorSaved(false);
      setStatus(`Showing training round ${roundIndex + 1}`);
    } catch (e) {
      setStatus(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function selectEvaluation(evaluationId: string) {
    const sessionId = sessionController.sessionId ?? run?.session_id;
    const roundIndex = selectedRound ?? run?.round_index ?? run?.segment;
    if (!sessionId || roundIndex === undefined || evaluationId === selectedEvaluation) return;
    setLoading(true);
    try {
      setRun(await loadRoundWithFinalization(sessionId, roundIndex, evaluationId));
      setSelectedEvaluation(evaluationId);
      setStatus(`Showing holdout evaluation ${evaluationId.slice(0, 8)}`);
    } catch (reason) {
      setStatus(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  }

  async function finalizeSelectedRound(strategyPlanId?: string) {
    const sessionId = sessionController.sessionId ?? run?.session_id;
    const roundIndex = selectedRound ?? run?.round_index ?? run?.segment;
    if (
      !sessionId ||
      roundIndex === undefined ||
      finalizing ||
      sessionController.finalizing
    ) return;

    setFinalizing(true);
    setStatus("Checking locked-holdout evidence history...");
    let submitted = false;
    try {
      const existing = await listSessionFinalizations(sessionId);
      const repeat = existing.some((item) => item.report_available);
      if (
        repeat &&
        !window.confirm(
          "This frozen holdout has already been viewed in this session. A further read is " +
            "exploratory and cannot be treated as fresh evidence. Finalize this round anyway?",
        )
      ) {
        setStatus("Holdout finalization cancelled");
        return;
      }

      const handle = await finalizeSessionRound(
        sessionId,
        roundIndex,
        repeat,
        strategyPlanId,
      );
      submitted = true;
      handledFinalizationRef.current = null;
      setSelectedEvaluation(handle.evaluation_id);
      setStatus(
        repeat
          ? "Running an exploratory repeated-holdout finalization..."
          : "Running the first locked-holdout finalization...",
      );
      // The application-level session controller owns polling. This survives tab
      // changes and reloads, then the completion effect above attaches the artifact.
      sessionController.refresh();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (!submitted) setFinalizing(false);
    }
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
        await sessionController.stop();
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
    setStatus(`Seeding a new session from ${ids.length} Formula Result(s)`);
  }

  function recoverFormulaDraft(index: number) {
    const recovered = recoveredFormulaDrafts[index];
    if (!recovered) return;
    const remaining = recoveredFormulaDrafts.filter((_, entryIndex) => entryIndex !== index);
    if (formulaDraft) {
      remaining.push({ label: "Previous active Formula Builder draft", draft: formulaDraft });
    }
    setFormulaDraft(recovered.draft);
    setRecoveredFormulaDrafts(remaining);
    setStatus(`Opened ${recovered.label}; the previous draft remains recoverable`);
  }

  async function openUniverseEditor(universeName: string, historyStart?: string) {
    try {
      const universe = await getUniverse(universeName);
      setUniverseDraft({
        name: universe.name,
        rows: rowsFromUniverse(universe),
        // Preserve the saved definition identity for bundled snapshots too. Price sync can then
        // use the compact universe request and its pinned aliases/fingerprint.
        selectedUniverse: universe.name,
        expectedStart: historyStart ?? defaultUniverseHistoryStart(universe),
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
        name: "best formula result",
        tree: parseFactor(run.best_factor),
        metrics: run.report
          ? {
              oos_ic: run.report.oos_ic,
              deflated_sharpe: run.report.deflated_sharpe,
            }
          : {
              validation_median_ic:
                run.selection?.median_oriented_ic ?? run.selection?.validation_fitness ?? 0,
              validation_objective:
                run.selection?.final_objective ?? run.selection?.validation_fitness ?? 0,
            },
        provenance: {
          session_id: run.session_id,
          cumulative_trials: run.cumulative_trials,
          test_reads: run.test_reads,
        },
      });
      setBestFactorSaved(true);
      setStatus("Saved Best Formula Result to the library");
    } catch (e) {
      setStatus(String(e));
    }
  }

  function openBestFactorCopy() {
    if (!run) return;
    const round = (run.round_index ?? run.segment ?? 0) + 1;
    setFormulaDraft({
      name: `best_formula_round_${round}_copy`,
      display_name: `Best Formula Result — Round ${round} copy`,
      description: "Editable copy of the validation-selected formula snapshot.",
      body: parseFactor(run.best_factor),
      inputs: [],
      arg_types: [],
      out_type: "signal",
      category: "custom",
      activeMode: "visual",
      loadedName: null,
      loadedRevision: null,
    });
    setExtendPage("formula");
    setTab("extend");
    setStatus(`Opened an editable copy of the Round ${round} formula`);
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

  // The agent works from a session: it needs the frozen split boundaries and the search history,
  // neither of which an ad-hoc expression has.
  const agentSessionId = sessionController.sessionId ?? run?.session_id ?? null;
  const agentRoundIndex = selectedRound ?? run?.round_index ?? run?.segment ?? null;

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
  if (run && !bestFactorSaved) {
    quitWarnings.push("The Best Formula Result isn't saved to your library.");
  }

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
        {tab !== "extend" && (
          <PageHeader
            // Defence in depth. `tab` is validated on the way in, so a miss here means a tab was
            // added without copy — which should look like a bare heading, not a white screen.
            title={PAGE_COPY[tab]?.title ?? tab}
            description={PAGE_COPY[tab]?.description ?? ""}
            actions={
              tab === "agent" ? (
                <span className="beta-badge" data-testid="agent-beta">
                  Beta
                </span>
              ) : undefined
            }
          />
        )}

        {error && <p className="error surface-message">{error}</p>}
        {loading && !run && <p className="surface-message">Loading...</p>}

        <ErrorBoundary key={tab}>
        {run && (tab === "dashboard" || tab === "factor" || tab === "genealogy") && (
          <RoundNavigator
            rounds={rounds}
            selectedRound={selectedRound}
            pending={searchRunning}
            onSelect={(roundIndex) => void selectRound(roundIndex)}
            onContinue={tab === "dashboard" && mode === "app" ? () => setTab("train") : undefined}
          />
        )}
        {run && tab === "dashboard" && evaluations.length > 0 && (
          <EvaluationNavigator
            evaluations={evaluations}
            selectedId={selectedEvaluation}
            onSelect={(evaluationId) => void selectEvaluation(evaluationId)}
          />
        )}
        {tab === "train" && (
          <section className="view-card" data-view="train">
            <div className="view-body">
              <TrainPanel
                seedIds={seedIds}
                controller={sessionController}
                onOpenDashboard={() => setTab("dashboard")}
                onOpenUniverseEditor={openUniverseEditor}
                onOpenDataSync={openUniverseEditor}
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
            <div className="view-body">
              <Dashboard
                report={run.report}
                history={run.history}
                extra={run}
                enableBenchmarks={mode === "app"}
                onFinalize={
                  mode === "app" && run.session_id && selectedRound !== null
                    ? (strategyPlanId) => void finalizeSelectedRound(strategyPlanId)
                    : undefined
                }
                finalizing={finalizing || sessionController.finalizing}
              />
            </div>
          </section>
        )}

        {run && factor && tab === "factor" && (
          <section className="view-card" data-view="factor">
            <div className="view-body">
              <BestFormulaResultPage
                factor={factor}
                canSave={mode === "app"}
                saved={bestFactorSaved}
                onSave={() => void saveBestFactor()}
                onOpenCopy={openBestFactorCopy}
                onLegacySelection={setSelectedNode}
              />
            </div>
          </section>
        )}

        {run && tab === "genealogy" && (
          <section className="view-card" data-view="genealogy">
            <div className="view-body">
              <Genealogy
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
            <div className="view-body">
              <LibraryPanel onSeed={startSeededSession} />
            </div>
          </section>
        )}

        {tab === "agent" && (
          <section className="view-card" data-view="agent">
            <div className="view-body">
              <AgentPage
                sessionId={mode === "app" ? (agentSessionId ?? undefined) : undefined}
                sessionName={sessionController.state?.name}
                roundIndex={agentRoundIndex ?? undefined}
                onRoundsChanged={() => {
                  const id = sessionController.sessionId ?? run?.session_id;
                  if (id) void listSessionRounds(id).then(setRounds).catch(() => {});
                }}
              />
            </div>
          </section>
        )}

        {tab === "extend" && (
          <section className="view-card" data-view="extend">
            <div className="view-body">
              <ExtendPanel
                page={extendPage}
                universeDraft={universeDraft}
                onUniverseDraftChange={setUniverseDraft}
                formulaDraft={formulaDraft}
                onFormulaDraftChange={setFormulaDraft}
                recoveredFormulaDrafts={recoveredFormulaDrafts}
                onRecoverFormulaDraft={recoverFormulaDraft}
                onOpenDataSync={(universeName) => {
                  if (universeName) {
                    void openUniverseEditor(universeName);
                  } else {
                    setExtendPage("universe");
                  }
                }}
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
                    Save Best Formula Result
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
