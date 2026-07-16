import { useEffect, useMemo, useState } from "react";
import { getTrainingCapabilities } from "../api/client";
import type {
  TrainingCapabilities,
  TrainingResourceProfile,
  TrainingResourcesRequest,
} from "../api/types";

const PROFILE_COPY: Record<
  TrainingResourceProfile,
  { label: string; description: string }
> = {
  light: { label: "Light", description: "keeps the computer responsive" },
  auto: { label: "Auto", description: "recommended" },
  maximum: { label: "Maximum", description: "fastest, high system load" },
  custom: { label: "Custom", description: "Choose 10–100%" },
};

function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GiB`;
  return `${Math.round(value / 1024 ** 2)} MiB`;
}

function profileWorkers(
  profile: TrainingResourceProfile,
  customPercent: number,
  capabilities: TrainingCapabilities | null,
): { percent: number; workers: number; memory: number } | null {
  if (!capabilities) return null;
  if (profile !== "custom") {
    const resolved = capabilities.profiles[profile];
    return {
      percent: resolved.percent,
      workers: resolved.workers,
      memory: resolved.run_memory_budget_bytes,
    };
  }
  const percent = Math.max(
    capabilities.cpu_budget_percent_min,
    Math.min(capabilities.cpu_budget_percent_max, customPercent),
  );
  const requested = Math.max(1, Math.ceil(capabilities.worker_capacity * percent / 100));
  const workers = capabilities.fallback_reason ? 1 : requested;
  const perWorker = Math.max(
    1,
    Math.floor(capabilities.memory_budget_bytes / capabilities.worker_capacity),
  );
  return { percent, workers, memory: perWorker * workers };
}

export function TrainingResourcePicker({
  value,
  onChange,
  disabled = false,
  label = "Computer resources",
}: {
  value: TrainingResourcesRequest;
  onChange: (value: TrainingResourcesRequest) => void;
  disabled?: boolean;
  label?: string;
}) {
  const [capabilities, setCapabilities] = useState<TrainingCapabilities | null>(null);

  useEffect(() => {
    try {
      getTrainingCapabilities().then(setCapabilities).catch(() => setCapabilities(null));
    } catch {
      // Older/demo clients may not expose capabilities; Auto remains a safe server default.
      setCapabilities(null);
    }
  }, []);

  const customPercent = value.cpu_budget_percent ?? 50;
  const resolved = useMemo(
    () => profileWorkers(value.profile, customPercent, capabilities),
    [value.profile, customPercent, capabilities],
  );

  return (
    <fieldset className="resource-picker" data-testid="resource-picker">
      <legend>{label}</legend>
      <label className="field">
        <span className="field-label">Resource profile</span>
        <select
          aria-label="Resource profile"
          value={value.profile}
          disabled={disabled}
          onChange={(event) => {
            const profile = event.target.value as TrainingResourceProfile;
            onChange({
              profile,
              cpu_budget_percent: profile === "custom" ? customPercent : null,
            });
          }}
        >
          {(Object.keys(PROFILE_COPY) as TrainingResourceProfile[]).map((profile) => {
            const resolvedProfile = capabilities && profile !== "custom"
              ? capabilities.profiles[profile]
              : null;
            const deviceRelative = resolvedProfile
              ? ` — ${resolvedProfile.percent}%, ${resolvedProfile.workers} of ${capabilities?.detected_cpus} CPUs · `
              : " — ";
            return (
              <option key={profile} value={profile}>
                {PROFILE_COPY[profile].label}{deviceRelative}{PROFILE_COPY[profile].description}
              </option>
            );
          })}
        </select>
      </label>

      {value.profile === "custom" && (
        <label className="field resource-custom">
          <span className="field-label">
            Custom CPU percentage <strong>{customPercent}%</strong>
          </span>
          <input
            type="range"
            min={capabilities?.cpu_budget_percent_min ?? 10}
            max={capabilities?.cpu_budget_percent_max ?? 100}
            step={5}
            value={customPercent}
            disabled={disabled}
            aria-label="Custom CPU percentage"
            onChange={(event) =>
              onChange({ profile: "custom", cpu_budget_percent: Number(event.target.value) })
            }
          />
        </label>
      )}

      {resolved && capabilities && (
        <p className="resource-summary" data-testid="resource-summary">
          Uses {resolved.percent}%: {resolved.workers} of {capabilities.detected_cpus} CPUs
          {capabilities.detected_cpus > capabilities.worker_capacity ? " (one reserved)" : ""}, with
          up to {formatBytes(resolved.memory)} working memory.
        </p>
      )}
      {capabilities?.fallback_reason && (
        <p className="resource-warning" role="status">
          {capabilities.fallback_reason}
        </p>
      )}
      {value.profile === "maximum" && !capabilities?.fallback_reason && (
        <p className="resource-warning" role="status">
          Maximum may make other applications less responsive while training runs.
        </p>
      )}
    </fieldset>
  );
}
