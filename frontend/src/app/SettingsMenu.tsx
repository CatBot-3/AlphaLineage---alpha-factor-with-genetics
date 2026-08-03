// Header settings dialog. Workspace actions, backend configuration, storage, and
// local-data cleanup live here; application shutdown remains a first-level header action.

import { useEffect, useId, useRef, useState } from "react";
import { clearData, getDataUsage, getSettings, putSettings } from "../api/client";
import type { AppMode, DataUsageRow, Settings } from "../api/types";
import { CompactSection } from "./CompactSection";

// Kept in step with `alphalineage.explain.providers.PROVIDERS`. Labels only — the backend is
// still the authority on which providers exist and what they may call.
const LLM_PROVIDERS = [
  { id: "openai", label: "OpenAI" },
  { id: "anthropic", label: "Anthropic" },
  { id: "deepseek", label: "DeepSeek" },
  { id: "openai_compatible", label: "OpenAI-compatible endpoint (custom)" },
];

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
}: {
  mode: AppMode;
  onRefreshRun: () => void;
  onSaveLocal: () => void;
  onLoadLocal: () => void;
  onSaveBackend: () => void;
  onLoadBackend: () => void;
}) {
  const backend = mode === "app";
  const [open, setOpen] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [usage, setUsage] = useState<DataUsageRow[]>([]);
  const [tiingoKey, setTiingoKey] = useState("");
  const [factorsDir, setFactorsDir] = useState("");
  const [llmProvider, setLlmProvider] = useState("openai");
  const [llmModel, setLlmModel] = useState("");
  const [llmBaseUrl, setLlmBaseUrl] = useState("");
  const [llmKey, setLlmKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const descriptionId = useId();
  const usageBytes = usage.reduce((sum, row) => sum + row.bytes, 0);

  async function refreshSettings() {
    if (!backend) return;
    setLoading(true);
    setError(null);
    const [settingsResult, usageResult] = await Promise.allSettled([
      getSettings(),
      getDataUsage(),
    ]);
    if (settingsResult.status === "fulfilled") {
      setSettings(settingsResult.value);
      setFactorsDir(settingsResult.value.factors_dir);
      setLlmProvider(settingsResult.value.llm_provider || "openai");
      setLlmModel(settingsResult.value.llm_model ?? "");
      setLlmBaseUrl(settingsResult.value.llm_base_url ?? "");
    } else {
      setSettings(null);
      setError("Settings could not be loaded. Check the backend connection and try again.");
    }
    if (usageResult.status === "fulfilled") {
      setUsage(usageResult.value);
    } else if (settingsResult.status === "fulfilled") {
      setError("Local data usage could not be loaded. Settings are still available.");
    }
    setLoading(false);
  }

  useEffect(() => {
    if (!open) return;
    if (backend) void refreshSettings();
    const frame = window.requestAnimationFrame(() => closeRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [open, backend]);

  useEffect(() => {
    if (!open) return;
    function closeAndRestoreFocus() {
      setOpen(false);
      window.requestAnimationFrame(() => triggerRef.current?.focus());
    }
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as globalThis.Node)) {
        closeAndRestoreFocus();
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") closeAndRestoreFocus();
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function closeAndRestoreFocus() {
    setOpen(false);
    window.requestAnimationFrame(() => triggerRef.current?.focus());
  }

  async function updateSettings(
    operation: string,
    update: Parameters<typeof putSettings>[0],
  ) {
    setSaving(operation);
    setError(null);
    try {
      const next = await putSettings(update);
      setSettings(next);
      setFactorsDir(next.factors_dir);
      return next;
    } catch {
      setError("The setting could not be saved. Nothing was changed.");
      return null;
    } finally {
      setSaving(null);
    }
  }

  async function saveEvaluator(evaluator: Settings["evaluator"]) {
    await updateSettings("evaluator", { evaluator });
  }

  async function saveTiingo() {
    if (!tiingoKey.trim()) return;
    const next = await updateSettings("tiingo", { tiingo_api_key: tiingoKey });
    if (next) setTiingoKey("");
  }

  async function removeStoredTiingo() {
    if (!window.confirm("Remove the stored Tiingo API key? Environment configuration is unaffected.")) {
      return;
    }
    await updateSettings("tiingo-remove", { tiingo_api_key: "" });
    setTiingoKey("");
  }

  async function saveFactorsDir() {
    await updateSettings("storage", { factors_dir: factorsDir });
  }

  async function saveLlm() {
    const next = await updateSettings("llm", {
      llm_provider: llmProvider,
      llm_model: llmModel,
      llm_base_url: llmBaseUrl,
      // Only send the key when one was typed, so saving a model change never clears it.
      ...(llmKey.trim() ? { llm_api_key: llmKey, llm_api_key_provider: llmProvider } : {}),
    });
    if (next) setLlmKey("");
  }

  async function removeLlmKey() {
    if (!window.confirm("Remove the stored explanation API key?")) return;
    await updateSettings("llm-remove", {
      llm_api_key: "",
      llm_api_key_provider: llmProvider,
    });
    setLlmKey("");
  }

  async function clearCategory(row: DataUsageRow) {
    if (!window.confirm(`Delete all ${row.label.toLowerCase()}? This cannot be undone.`)) return;
    setSaving(`clear-${row.key}`);
    setError(null);
    try {
      await clearData(row.key);
      setUsage(await getDataUsage());
    } catch {
      setError(`${row.label} could not be cleared.`);
    } finally {
      setSaving(null);
    }
  }

  const keySource =
    settings?.tiingo_api_key_source ??
    (settings?.tiingo_api_key_set ? "stored" : "none");
  const storedKeySet =
    settings?.tiingo_stored_key_set ??
    (keySource === "stored" && Boolean(settings?.tiingo_api_key_set));
  const keySummary =
    keySource === "environment"
      ? "Key: environment"
      : storedKeySet
        ? "Key: stored"
        : "Key: not configured";

  const llmKeySource = settings?.llm_api_key_source ?? "none";
  const llmSummary = settings?.llm_configured
    ? `${settings.llm_provider} / ${settings.llm_model}`
    : "Not configured";

  return (
    <div className="settings-menu" ref={ref}>
      <button
        ref={triggerRef}
        type="button"
        className="gear"
        aria-label="Open settings"
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Settings"
        onClick={() => {
          if (open) closeAndRestoreFocus();
          else setOpen(true);
        }}
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          width="20"
          height="20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.83 2.83-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V21h-4v-.08A1.7 1.7 0 0 0 8.97 19.4a1.7 1.7 0 0 0-1.88.34l-.06.06-2.83-2.83.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.56-1.03H3v-4h.08A1.7 1.7 0 0 0 4.6 8.94a1.7 1.7 0 0 0-.34-1.88L4.2 7l2.83-2.83.06.06a1.7 1.7 0 0 0 1.88.34A1.7 1.7 0 0 0 10 3.01V3h4v.08a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.88-.34l.06-.06L19.8 7l-.06.06a1.7 1.7 0 0 0-.34 1.88 1.7 1.7 0 0 0 1.56 1.03H21v4h-.08A1.7 1.7 0 0 0 19.4 15Z" />
        </svg>
      </button>

      {open && (
        <div
          className="settings-popover"
          role="dialog"
          aria-modal="false"
          aria-labelledby={titleId}
          aria-describedby={descriptionId}
          data-testid="settings-popover"
        >
          <header className="settings-popover__head">
            <div>
              <h2 id={titleId}>Settings</h2>
              <p id={descriptionId}>Workspace, data access, and local storage.</p>
            </div>
            <button
              ref={closeRef}
              type="button"
              className="settings-popover__close"
              aria-label="Close settings"
              onClick={closeAndRestoreFocus}
            >
              <span aria-hidden="true">×</span>
            </button>
          </header>

          {error && (
            <div className="settings-feedback settings-feedback--error" role="alert">
              <span>{error}</span>
              {backend && (
                <button type="button" onClick={() => void refreshSettings()}>
                  Retry
                </button>
              )}
            </div>
          )}
          {loading && (
            <div className="settings-feedback" role="status" aria-live="polite">
              Refreshing backend settings…
            </div>
          )}

          <CompactSection
            title="Workspace"
            summary={backend ? "Backend connected" : "Static demo"}
            className="settings-section"
            defaultOpen
          >
            <div className="settings-workspace-actions">
              <button
                type="button"
                className="settings-action settings-action--primary"
                onClick={onRefreshRun}
              >
                {backend ? "Run search" : "Load demo"}
              </button>
              <button type="button" className="settings-action" onClick={onSaveLocal}>
                Save local
              </button>
              <button type="button" className="settings-action" onClick={onLoadLocal}>
                Load local
              </button>
              <button
                type="button"
                className="settings-action"
                onClick={onSaveBackend}
                disabled={!backend}
              >
                Save backend
              </button>
              <button
                type="button"
                className="settings-action"
                onClick={onLoadBackend}
                disabled={!backend}
              >
                Load backend
              </button>
            </div>
          </CompactSection>

          <CompactSection
            title="Data & evaluator"
            summary={backend && settings ? keySummary : "Backend only"}
            className="settings-section"
          >
            {backend && settings ? (
              <>
                <div className="settings-control settings-key">
                  <div className="settings-control__label-row">
                    <span className="settings-control__label">Tiingo API key</span>
                    <span className={`settings-status settings-status--${keySource}`}>
                      {keySource === "environment"
                        ? "Environment"
                        : storedKeySet
                          ? "Stored"
                          : "Not configured"}
                    </span>
                  </div>
                  {keySource === "environment" ? (
                    <>
                      <p className="settings-help">
                        Configured through the process environment or <code>.env</code>. Edit that
                        source and restart AlphaLineage to change it.
                      </p>
                      {storedKeySet && (
                        <>
                          <p className="settings-help">
                            A separate stored fallback key also exists but is not currently used.
                          </p>
                          <button
                            type="button"
                            className="settings-action settings-action--danger"
                            disabled={saving === "tiingo-remove"}
                            onClick={() => void removeStoredTiingo()}
                          >
                            {saving === "tiingo-remove" ? "Removing…" : "Remove stored key"}
                          </button>
                        </>
                      )}
                    </>
                  ) : (
                    <>
                      <label className="settings-control">
                        <span className="settings-control__label">
                          {storedKeySet ? "Replace stored key" : "Store a key"}
                        </span>
                        <input
                          type="password"
                          aria-label="Tiingo API key"
                          autoComplete="off"
                          placeholder="Paste key"
                          value={tiingoKey}
                          onChange={(e) => setTiingoKey(e.target.value)}
                        />
                      </label>
                      <div className="settings-inline-actions">
                        <button
                          type="button"
                          className="settings-action"
                          disabled={!tiingoKey.trim() || saving === "tiingo"}
                          onClick={() => void saveTiingo()}
                        >
                          {saving === "tiingo" ? "Saving…" : "Save key"}
                        </button>
                        {storedKeySet && (
                          <button
                            type="button"
                            className="settings-action settings-action--danger"
                            disabled={saving === "tiingo-remove"}
                            onClick={() => void removeStoredTiingo()}
                          >
                            {saving === "tiingo-remove" ? "Removing…" : "Remove stored key"}
                          </button>
                        )}
                      </div>
                    </>
                  )}
                </div>

                <label className="settings-control">
                  <span className="settings-control__label">Evaluator backend</span>
                  <select
                    aria-label="Evaluator backend"
                    value={settings.evaluator}
                    disabled={saving === "evaluator"}
                    onChange={(e) => void saveEvaluator(e.target.value as Settings["evaluator"])}
                  >
                    <option value="auto">auto</option>
                    <option value="python">python</option>
                    <option value="cpp" disabled={!settings.cpp_available}>
                      cpp{settings.cpp_available ? "" : " (not built)"}
                    </option>
                  </select>
                </label>
              </>
            ) : (
              <p className="panel-note">
                {backend && loading
                  ? "Loading settings..."
                  : backend
                    ? "Settings are unavailable until the backend reconnects."
                    : "Backend settings are available when the local backend is running."}
              </p>
            )}
          </CompactSection>

          <CompactSection
            title="Explanation model"
            summary={backend && settings ? llmSummary : "Backend only"}
            className="settings-section"
          >
            {backend && settings ? (
              <>
                <p className="settings-help">
                  Optional. Used only to explain a discovered factor and suggest changes; it
                  never runs a search or touches your data. The key is sent to the provider you
                  pick and is never written into a saved explanation.
                </p>
                <label className="settings-control">
                  <span className="settings-control__label">Provider</span>
                  <select
                    aria-label="Explanation provider"
                    value={llmProvider}
                    onChange={(e) => setLlmProvider(e.target.value)}
                  >
                    {LLM_PROVIDERS.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="settings-control">
                  <span className="settings-control__label">Model</span>
                  <input
                    aria-label="Explanation model"
                    value={llmModel}
                    placeholder="e.g. gpt-4o"
                    onChange={(e) => setLlmModel(e.target.value)}
                  />
                </label>
                {llmProvider === "openai_compatible" && (
                  <label className="settings-control">
                    <span className="settings-control__label">Base URL</span>
                    <input
                      aria-label="Explanation base URL"
                      value={llmBaseUrl}
                      placeholder="https://host/v1"
                      onChange={(e) => setLlmBaseUrl(e.target.value)}
                    />
                  </label>
                )}
                <div className="settings-control settings-key">
                  <div className="settings-control__label-row">
                    <span className="settings-control__label">API key</span>
                    <span className={`settings-status settings-status--${llmKeySource}`}>
                      {llmKeySource === "environment"
                        ? "Environment"
                        : llmKeySource === "stored"
                          ? "Stored"
                          : "Not configured"}
                    </span>
                  </div>
                  {llmKeySource === "environment" ? (
                    <p className="settings-help">
                      Provided through the process environment. Edit that source and restart
                      AlphaLineage to change it.
                    </p>
                  ) : (
                    <label className="settings-control">
                      <input
                        type="password"
                        aria-label="Explanation API key"
                        value={llmKey}
                        placeholder="paste a key"
                        onChange={(e) => setLlmKey(e.target.value)}
                      />
                    </label>
                  )}
                </div>
                <div className="settings-workspace-actions">
                  <button
                    type="button"
                    className="settings-action settings-action--primary"
                    disabled={saving === "llm"}
                    onClick={() => void saveLlm()}
                  >
                    {saving === "llm" ? "Saving…" : "Save"}
                  </button>
                  {settings.llm_api_key_set && llmKeySource === "stored" && (
                    <button
                      type="button"
                      className="settings-action settings-action--danger"
                      disabled={saving === "llm-remove"}
                      onClick={() => void removeLlmKey()}
                    >
                      {saving === "llm-remove" ? "Removing…" : "Remove stored key"}
                    </button>
                  )}
                </div>
              </>
            ) : (
              <p className="panel-note">An explanation model requires the local backend.</p>
            )}
          </CompactSection>

          <CompactSection
            title="Storage"
            summary={backend ? "Formula Results" : "Backend only"}
            className="settings-section"
          >
            {backend && settings ? (
              <>
                <label className="settings-control">
                  <span className="settings-control__label">Formula Results directory</span>
                  <input
                    aria-label="Formula Results directory"
                    value={factorsDir}
                    onChange={(e) => setFactorsDir(e.target.value)}
                  />
                </label>
                <button
                  type="button"
                  className="settings-action"
                  disabled={!factorsDir.trim() || saving === "storage"}
                  onClick={() => void saveFactorsDir()}
                >
                  {saving === "storage" ? "Saving…" : "Save folder"}
                </button>
              </>
            ) : (
              <p className="panel-note">Storage settings require the local backend.</p>
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
                      className="settings-action"
                      disabled={row.count === 0 || saving === `clear-${row.key}`}
                      onClick={() => void clearCategory(row)}
                    >
                      {saving === `clear-${row.key}` ? "Clearing…" : "Clear"}
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="panel-note">Cached data can be reviewed after connecting the backend.</p>
            )}
          </CompactSection>
        </div>
      )}
    </div>
  );
}
