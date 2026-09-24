import type { ExecutionTiming, GpConfig } from "../api/types";

// A light, interactive preset (the full dev.yaml defaults are 200x25; that is too slow for a
// click-and-watch first run). Every field is editable in the form, so users can scale up.
export const DEFAULT_CONFIG: GpConfig = {
  novelty_mode: "balanced",
  population_size: 80,
  generations: 12,
  tournament_size: 3,
  crossover_rate: 0.8,
  subtree_mutation_rate: 0.1,
  point_mutation_rate: 0.1,
  max_depth: 6,
  max_nodes: 40,
  parsimony: 0.001,
  complexity_penalty_mode: "normalized_budget",
  complexity_penalty_value: 0.005,
  parameter_neighbor_fraction: 0.25,
  validation_folds: 3,
  exploration_profile: "aggressive",
  elitism: 1,
  ic_method: "spearman",
  min_names: 5,
  horizon: 1,
  execution: "next_open",
  min_depth: 2,
  seed: 0,
  time_budget_s: null,
};

// Fields exposed as the "core" knobs; the rest live behind an Advanced disclosure.
export const CORE_FIELDS: Array<{ key: keyof GpConfig; label: string }> = [
  { key: "population_size", label: "Population" },
  { key: "generations", label: "Generations" },

];

export const ADVANCED_FIELDS: Array<{ key: keyof GpConfig; label: string }> = [
  { key: "max_depth", label: "Max depth" },
  { key: "max_nodes", label: "Max nodes" },
  { key: "seed", label: "Seed" },
  { key: "tournament_size", label: "Tournament size" },
  { key: "crossover_rate", label: "Crossover rate" },
  { key: "subtree_mutation_rate", label: "Subtree mutation" },
  { key: "point_mutation_rate", label: "Point mutation" },
  { key: "complexity_penalty_value", label: "Maximum complexity deduction" },
  { key: "parameter_neighbor_fraction", label: "Parameter-neighbor fraction" },
  { key: "validation_folds", label: "Validation folds" },
  { key: "elitism", label: "Elitism" },
  { key: "min_names", label: "Min names" },
  { key: "horizon", label: "Horizon" },
];

export const EXECUTION_TIMING_OPTIONS: Array<{
  value: ExecutionTiming;
  label: string;
  short: string;
  description: string;
}> = [
  {
    value: "next_open",
    label: "Next day's open (recommended)",
    short: "Next open",
    description:
      "Signals are computed after today's close and traded at tomorrow's opening auction, " +
      "then held open to open. Matches an end-of-day workflow. The open is a noisy, wide-spread " +
      "price, so keep costs conservative.",
  },
  {
    value: "next_close",
    label: "Next day's close (one-day delay)",
    short: "Next close",
    description:
      "Signals are traded in tomorrow's closing auction and held close to close. The most " +
      "conservative choice: anything that survives a full day's delay is not a same-day artifact.",
  },
  {
    value: "close",
    label: "Same close (optimistic, legacy)",
    short: "Same close",
    description:
      "Assumes you can trade at the very close that produced the signal. Useful for comparing " +
      "with older sessions, but it overstates fast signals, which often feed on closing-price " +
      "noise that reverses overnight.",
  },
];

export function executionTimingLabel(value: ExecutionTiming | undefined | null): string {
  const resolved = value ?? "close";
  return EXECUTION_TIMING_OPTIONS.find((option) => option.value === resolved)?.short ?? resolved;
}
