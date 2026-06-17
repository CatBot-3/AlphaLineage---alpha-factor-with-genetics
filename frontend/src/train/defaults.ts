import type { GpConfig } from "../api/types";

// A light, interactive preset (the full dev.yaml defaults are 200x25; that is too slow for a
// click-and-watch first run). Every field is editable in the form, so users can scale up.
export const DEFAULT_CONFIG: GpConfig = {
  population_size: 80,
  generations: 12,
  tournament_size: 3,
  crossover_rate: 0.8,
  subtree_mutation_rate: 0.1,
  point_mutation_rate: 0.1,
  max_depth: 6,
  max_nodes: 40,
  parsimony: 0.001,
  elitism: 1,
  ic_method: "spearman",
  min_names: 5,
  horizon: 1,
  min_depth: 2,
  seed: 0,
  time_budget_s: null,
};

// Fields exposed as the "core" knobs; the rest live behind an Advanced disclosure.
export const CORE_FIELDS: Array<{ key: keyof GpConfig; label: string }> = [
  { key: "population_size", label: "Population" },
  { key: "generations", label: "Generations" },
  { key: "max_depth", label: "Max depth" },
  { key: "max_nodes", label: "Max nodes" },
  { key: "seed", label: "Seed" },
];

export const ADVANCED_FIELDS: Array<{ key: keyof GpConfig; label: string }> = [
  { key: "tournament_size", label: "Tournament size" },
  { key: "crossover_rate", label: "Crossover rate" },
  { key: "subtree_mutation_rate", label: "Subtree mutation" },
  { key: "point_mutation_rate", label: "Point mutation" },
  { key: "parsimony", label: "Parsimony" },
  { key: "elitism", label: "Elitism" },
  { key: "min_names", label: "Min names" },
  { key: "horizon", label: "Horizon" },
];
