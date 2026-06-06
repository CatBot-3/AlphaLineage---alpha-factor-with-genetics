import { useEffect, useState } from "react";
import { loadRun } from "./api/dataSource";
import { parseFactor, type RunResult } from "./api/types";
import { Dashboard } from "./dashboard/Dashboard";
import { FactorDetail } from "./factor/FactorDetail";
import { FactorTree } from "./factor/FactorTree";
import type { TreeNodeData } from "./factor/treeToFlow";
import { Genealogy } from "./genealogy/Genealogy";
import { LineageDetail } from "./genealogy/LineageDetail";

type Tab = "dashboard" | "factor" | "genealogy";

export function App() {
  const [run, setRun] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("dashboard");
  const [selectedNode, setSelectedNode] = useState<TreeNodeData | null>(null);
  const [selectedLineage, setSelectedLineage] = useState<number | null>(null);

  useEffect(() => {
    loadRun()
      .then(setRun)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="app">
        <h1>AlphaForge</h1>
        <p className="error">{error}</p>
      </div>
    );
  }
  if (!run) {
    return (
      <div className="app">
        <h1>AlphaForge</h1>
        <p>Loading…</p>
      </div>
    );
  }

  const factor = parseFactor(run.best_factor);
  return (
    <div className="app">
      <header>
        <h1>AlphaForge</h1>
        <p className="tagline">Honest, deflated alpha mining</p>
      </header>
      <nav>
        <button onClick={() => setTab("dashboard")} aria-current={tab === "dashboard"}>
          Metrics
        </button>
        <button onClick={() => setTab("factor")} aria-current={tab === "factor"}>
          Best factor
        </button>
        <button onClick={() => setTab("genealogy")} aria-current={tab === "genealogy"}>
          Genealogy
        </button>
      </nav>
      <main>
        {tab === "dashboard" && <Dashboard report={run.report} history={run.history} />}
        {tab === "factor" && (
          <div className="split">
            <FactorTree factor={factor} onSelect={setSelectedNode} />
            <FactorDetail node={selectedNode} />
          </div>
        )}
        {tab === "genealogy" && (
          <div className="split">
            <Genealogy lineage={run.lineage} onSelect={setSelectedLineage} />
            <LineageDetail
              lineage={run.lineage}
              selectedId={selectedLineage}
              onSelect={setSelectedLineage}
            />
          </div>
        )}
      </main>
      <footer>Not investment advice. Research output only.</footer>
    </div>
  );
}
