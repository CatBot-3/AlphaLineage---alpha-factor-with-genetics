import { useCallback, useEffect, useState } from "react";
import {
  getWorkspace,
  listWorkspaces,
  saveWorkspace,
} from "../api/client";
import { loadRun } from "../api/dataSource";
import {
  parseFactor,
  type OperatorComposerDraft,
  type RunResult,
  type UniverseDraft,
  type WorkspaceSnapshot,
} from "../api/types";
import { Dashboard } from "../dashboard/Dashboard";
import { OperatorComposer } from "../extend/OperatorComposer";
import { UniverseBuilder } from "../extend/UniverseBuilder";
import { FactorDetail } from "../factor/FactorDetail";
import { FactorTree } from "../factor/FactorTree";
import type { TreeNodeData } from "../factor/treeToFlow";
import { Genealogy } from "../genealogy/Genealogy";
import { LineageDetail } from "../genealogy/LineageDetail";
import { AppShell, type Tab } from "./AppShell";
import { getAppMode } from "./mode";
import {
  makeWorkspaceSnapshot,
  readLocalWorkspace,
  writeLocalWorkspace,
} from "./workspace";

function tabFromWorkspace(snapshot: WorkspaceSnapshot | null): Tab {
  return snapshot?.ui.selectedTab ?? "dashboard";
}

function applyNode(node?: { name: string; value?: number } | null): TreeNodeData | null {
  return node ? { name: node.name, value: node.value } : null;
}

export function App() {
  const mode = getAppMode();
  const [initialWorkspace] = useState(() => readLocalWorkspace());
  const [run, setRun] = useState<RunResult | null>(initialWorkspace?.run ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!initialWorkspace?.run);
  const [tab, setTab] = useState<Tab>(tabFromWorkspace(initialWorkspace));
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
  const [status, setStatus] = useState<string | null>(
    initialWorkspace?.run ? "Loaded local workspace" : null,
  );

  const currentSnapshot = useCallback(
    () =>
      makeWorkspaceSnapshot({
        run,
        universeDraft,
        operatorDraft,
        ui: {
          selectedTab: tab,
          selectedFactorNode: selectedNode
            ? { name: selectedNode.name, value: selectedNode.value }
            : null,
          selectedLineage,
        },
      }),
    [operatorDraft, run, selectedLineage, selectedNode, tab, universeDraft],
  );

  const applyWorkspace = useCallback((snapshot: WorkspaceSnapshot) => {
    setRun(snapshot.run);
    setTab(snapshot.ui.selectedTab ?? "dashboard");
    setSelectedNode(applyNode(snapshot.ui.selectedFactorNode));
    setSelectedLineage(snapshot.ui.selectedLineage ?? null);
    setUniverseDraft(snapshot.universeDraft);
    setOperatorDraft(snapshot.operatorDraft);
    setStatus(`Loaded ${snapshot.name}`);
  }, []);

  const refreshRun = useCallback(() => {
    setLoading(true);
    setError(null);
    setStatus(mode === "app" ? "Running backend search..." : "Loading static demo...");
    loadRun()
      .then((result) => {
        setRun(result);
        setStatus(mode === "app" ? "Backend run completed" : "Static demo loaded");
      })
      .catch((e) => {
        setError(String(e));
        setStatus("Load failed");
      })
      .finally(() => setLoading(false));
  }, [mode]);

  useEffect(() => {
    if (!initialWorkspace?.run) {
      refreshRun();
    }
  }, [initialWorkspace?.run, refreshRun]);

  useEffect(() => {
    writeLocalWorkspace(currentSnapshot());
  }, [currentSnapshot]);

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
      const saved = await saveWorkspace({ ...currentSnapshot(), id: undefined, name: "Backend Workspace" });
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

  return (
    <AppShell
      mode={mode}
      tab={tab}
      status={status}
      onTabChange={setTab}
      onRefreshRun={refreshRun}
      onSaveLocal={saveLocal}
      onLoadLocal={loadLocal}
      onSaveBackend={saveBackend}
      onLoadBackend={loadBackend}
    >
      <section className="app-page">
        <header className="view-head">
          <div className="view-tag">
            <span className="view-tag__mark" aria-hidden="true" />
            <span>#ALPHALINEAGE</span>
          </div>
          <h1>Honest factor evolution</h1>
          <p>
            Trace each generated alpha from metrics to tree structure to genetic lineage.
          </p>
          <div className="view-rule" aria-hidden="true" />
        </header>

        {error && <p className="error surface-message">{error}</p>}
        {loading && !run && <p className="surface-message">Loading...</p>}

        {run && tab === "dashboard" && (
          <section className="view-card" data-view="dashboard">
            <div className="view-rail">
              <span>01</span>
              <span />
            </div>
            <div className="view-body">
              <Dashboard report={run.report} history={run.history} />
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
              <FactorDetail node={selectedNode} />
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
              />
            </div>
          </section>
        )}

        {tab === "extend" && (
          <section className="view-card" data-view="extend">
            <div className="view-rail">
              <span>04</span>
              <span />
            </div>
            <div className="view-body extend">
              <UniverseBuilder
                draft={universeDraft}
                onDraftChange={setUniverseDraft}
                canSubmit={mode === "app"}
              />
              <OperatorComposer
                draft={operatorDraft}
                onDraftChange={setOperatorDraft}
                canSubmit={mode === "app"}
              />
            </div>
          </section>
        )}
      </section>
    </AppShell>
  );
}
