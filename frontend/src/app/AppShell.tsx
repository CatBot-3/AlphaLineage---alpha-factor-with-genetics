import type { AppMode, SyncProgressSnapshot } from "../api/types";
import type { ReactNode } from "react";
import type { ExtendPage } from "../extend/ExtendPanel";
import { DataPullProgress } from "./DataPullProgress";
import { ExtendMenu } from "./ExtendMenu";
import { SettingsMenu } from "./SettingsMenu";

export type Tab = "train" | "dashboard" | "factor" | "genealogy" | "library" | "extend";

const TABS: Array<{ id: Tab; label: string; backendOnly?: boolean }> = [
  { id: "train", label: "Train", backendOnly: true },
  { id: "dashboard", label: "Metrics" },
  { id: "factor", label: "Best factor" },
  { id: "genealogy", label: "Genealogy" },
  { id: "library", label: "Library", backendOnly: true },
  { id: "extend", label: "Extend" },
];

export function AppShell({
  mode,
  tab,
  status,
  progress,
  onTabChange,
  onRefreshRun,
  onSaveLocal,
  onLoadLocal,
  onSaveBackend,
  onLoadBackend,
  onQuit,
  onSelectExtendPage,
  children,
}: {
  mode: AppMode;
  tab: Tab;
  status?: string | null;
  progress?: SyncProgressSnapshot | null;
  onTabChange: (tab: Tab) => void;
  onRefreshRun: () => void;
  onSaveLocal: () => void;
  onLoadLocal: () => void;
  onSaveBackend: () => void;
  onLoadBackend: () => void;
  onQuit: () => void;
  onSelectExtendPage?: (page: ExtendPage) => void;
  children: ReactNode;
}) {
  const backend = mode === "app";
  return (
    <div className="app-shell" data-testid="app-shell">
      <header className="site-header">
        <nav className="nav" aria-label="Primary">
          <button className="nav__brand" type="button" onClick={() => onTabChange("dashboard")}>
            <span className="nav__brand-mark" aria-hidden="true" />
            <span>AlphaLineage</span>
          </button>
          <span className="nav__divider" aria-hidden="true" />
          <div className="nav__list" data-testid="main-nav">
            {TABS.map((item) =>
              item.id === "extend" ? (
                <ExtendMenu
                  key={item.id}
                  current={tab === "extend"}
                  disabled={item.backendOnly && !backend}
                  onSelect={(page) => onSelectExtendPage?.(page)}
                />
              ) : (
                <button
                  key={item.id}
                  className="nav__link"
                  type="button"
                  aria-current={tab === item.id ? "page" : undefined}
                  aria-disabled={item.backendOnly && !backend ? "true" : undefined}
                  onClick={() => onTabChange(item.id)}
                  title={
                    item.backendOnly && !backend
                      ? "Available as a locally saved draft in static demo mode"
                      : undefined
                  }
                >
                  {item.label}
                </button>
              ),
            )}
          </div>
        </nav>
        <div className="workspace-tools" aria-label="Workspace">
          <SettingsMenu
            mode={mode}
            onRefreshRun={onRefreshRun}
            onSaveLocal={onSaveLocal}
            onLoadLocal={onLoadLocal}
            onSaveBackend={onSaveBackend}
            onLoadBackend={onLoadBackend}
            onQuit={onQuit}
          />
        </div>
      </header>

      <main className="site-main">{children}</main>

      <DataPullProgress progress={progress ?? null} />

      <footer className="site-footer">
        <span>{status ?? "Not investment advice. Research output only."}</span>
        <span className="site-footer__rule" aria-hidden="true" />
        <span>AlphaLineage</span>
      </footer>
    </div>
  );
}
