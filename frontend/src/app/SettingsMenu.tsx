// The settings menu in the header. Absorbs workspace save/load actions, backend settings,
// local-data cleanup, and Quit. The sections intentionally reopen collapsed.

import { useEffect, useRef, useState } from "react";
import { clearData, getDataUsage, getSettings, putSettings } from "../api/client";
import type { AppMode, DataUsageRow, Settings } from "../api/types";
import { CompactSection } from "./CompactSection";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}

export function SettingsMenu({
  mode,
  onRefreshRun,
  onSaveLocal,
  onLoadLocal,
  onSaveBackend,
  onLoadBackend,
  onQuit,
}: {
  mode: AppMode;
  onRefreshRun: () => void;
  onSaveLocal: () => void;
  onLoadLocal: () => void;
  onSaveBackend: () => void;
  onLoadBackend: () => void;
  onQuit: () => void;
}) {
  const backend = mode === "app";
  const [open, setOpen] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [usage, setUsage] = useState<DataUsageRow[]>([]);
  const [tiingoKey, setTiingoKey] = useState("");
  const [factorsDir, setFactorsDir] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const usageBytes = usage.reduce((sum, row) => sum + row.bytes, 0);

  function refreshSettings() {
    getSettings()
      .then((s) => {
        setSettings(s);
        setFactorsDir(s.factors_dir);
      })
      .catch(() => undefined);
    getDataUsage().then(setUsage).catch(() => undefined);
  }

  useEffect(() => {
    if (open && backend && settings === null) refreshSettings();
  }, [open, backend, settings]);

  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as globalThis.Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  async function saveEvaluator(evaluator: Settings["evaluator"]) {
    setSettings(await putSettings({ evaluator }));
  }

  async function saveTiingo() {
    setSettings(await putSettings({ tiingo_api_key: tiingoKey }));
    setTiingoKey("");
  }

  async function saveFactorsDir() {
    setSettings(await putSettings({ factors_dir: factorsDir }));
  }

  async function clearCategory(row: DataUsageRow) {
    if (!window.confirm(`Delete all ${row.label.toLowerCase()}? This cannot be undone.`)) return;
    await clearData(row.key);
    getDataUsage().then(setUsage).catch(() => undefined);
  }

  return (
    <div className="settings-menu" ref={ref}>
      <button
        type="button"
        className="gear"
        aria-label="Settings menu"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => setOpen((v) => !v)}
      >
        Settings
      </button>

      {open && (
        <div className="settings-popover" role="menu" data-testid="settings-popover">
          <CompactSection
            title="Workspace"
            summary={backend ? "Backend connected" : "Static demo"}
            className="settings-section"
          >
            <button type="button" onClick={onRefreshRun}>
              {backend ? "Run search" : "Load demo"}
            </button>
            <button type="button" onClick={onSaveLocal}>
              Save local
            </button>
            <button type="button" onClick={onLoadLocal}>
              Load local
            </button>
            <button type="button" onClick={onSaveBackend} disabled={!backend}>
              Save backend
            </button>
            <button type="button" onClick={onLoadBackend} disabled={!backend}>
              Load backend
            </button>
          </CompactSection>

          <CompactSection
            title="Settings"
            summary={backend ? "Data keys and evaluator" : "Backend only"}
            className="settings-section"
          >
            {backend && settings ? (
              <>
                <label className="field">
                  <span className="field-label">
                    Tiingo API key {settings.tiingo_api_key_set ? "(set)" : "(not set)"}
                  </span>
                  <input
                    type="password"
                    aria-label="Tiingo API key"
                    placeholder="paste key"
                    value={tiingoKey}
                    onChange={(e) => setTiingoKey(e.target.value)}
                  />
                </label>
                <button type="button" className="ghost" onClick={saveTiingo}>
                  Save key
                </button>

                <label className="field">
                  <span className="field-label">Evaluator backend</span>
                  <select
                    aria-label="Evaluator backend"
                    value={settings.evaluator}
                    onChange={(e) => saveEvaluator(e.target.value as Settings["evaluator"])}
                  >
                    <option value="auto">auto</option>
                    <option value="python">python</option>
                    <option value="cpp" disabled={!settings.cpp_available}>
                      cpp{settings.cpp_available ? "" : " (not built)"}
                    </option>
                  </select>
                </label>

                <label className="field">
                  <span className="field-label">Factors directory</span>
                  <input
                    aria-label="Factors directory"
                    value={factorsDir}
                    onChange={(e) => setFactorsDir(e.target.value)}
                  />
                </label>
                <button type="button" className="ghost" onClick={saveFactorsDir}>
                  Save folder
                </button>
              </>
            ) : (
              <p className="panel-note">
                {backend
                  ? "Loading settings..."
                  : "Backend settings are available when the local backend is running."}
              </p>
            )}
          </CompactSection>

          <CompactSection
            title="Local data"
            summary={backend ? `${formatBytes(usageBytes)} cached` : "Backend only"}
            className="settings-section"
          >
            {backend ? (
              <ul className="data-rows" data-testid="data-rows">
                {usage.map((row) => (
                  <li key={row.key} className="data-row">
                    <span>
                      {row.label} - {formatBytes(row.bytes)} ({row.count})
                    </span>
                    <button
                      type="button"
                      className="ghost"
                      disabled={row.count === 0}
                      onClick={() => clearCategory(row)}
                    >
                      Clear
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="panel-note">Cached data can be reviewed after connecting the backend.</p>
            )}
          </CompactSection>

          <CompactSection
            title="Quit"
            summary={backend ? "Stop local servers" : "Backend only"}
            className="settings-section"
          >
            {backend ? (
              <button
                type="button"
                className="quit-action"
                data-testid="quit"
                onClick={() => {
                  setOpen(false);
                  onQuit();
                }}
              >
                Quit AlphaLineage
              </button>
            ) : (
              <p className="panel-note">The static demo has no local backend process to stop.</p>
            )}
          </CompactSection>
        </div>
      )}
    </div>
  );
}
