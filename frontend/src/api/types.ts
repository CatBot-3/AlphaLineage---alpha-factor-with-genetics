// TypeScript mirrors of the backend JSON shapes (core/tree.py, library/store.py, pipeline.py).

export interface FactorNode {
  name: string;
  value?: number;
  children?: FactorNode[];
}

export interface Report {
  oos_ic: number;
  deflated_sharpe: number;
  pbo: number;
  train_ic: number;
  n_trials: number;
  significant: boolean;
}

export interface LineageNode {
  id: number;
  generation: number;
  op: string;
  parents: number[];
  tree: FactorNode;
}

export interface Lineage {
  run_id: string;
  metadata: Record<string, unknown>;
  nodes: LineageNode[];
}

export interface HistoryPoint {
  generation: number;
  best_fitness: number;
  mean_fitness: number;
  best_ic: number;
}

export interface RunResult {
  best_factor: string | FactorNode; // the service returns a JSON string; demo JSON may store a tree
  report: Report;
  generations: number;
  history: HistoryPoint[];
  lineage: Lineage;
}

export function parseFactor(factor: string | FactorNode): FactorNode {
  return typeof factor === "string" ? (JSON.parse(factor) as FactorNode) : factor;
}
