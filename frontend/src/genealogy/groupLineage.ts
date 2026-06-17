// B4 (pure): collapse a lineage into per-generation groups keyed by evolution method, each
// group's members sorted by fitness descending. This is the data behind the grouped list view,
// which scales to thousands of nodes where a single DAG does not.

import type { Lineage, LineageNode } from "../api/types";

export interface GroupMember {
  id: number;
  fitness: number | null;
  op: string;
}

export interface MethodGroup {
  method: string;
  members: GroupMember[];
  count: number;
  bestFitness: number | null;
}

export interface GenerationGroup {
  generation: number;
  bestFitness: number | null;
  groups: MethodGroup[];
}

// The single most informative label for how an individual was made. Recombination dominates a
// composite op like "crossover+point_mut"; otherwise the mutation that touched it is the story.
export function primaryMethod(op: string): string {
  if (op.includes("elite")) return "elite";
  if (op.includes("seed")) return "seed";
  if (op.includes("init")) return "init";
  if (op.includes("crossover")) return "crossover";
  if (op.includes("subtree_mut")) return "subtree mutation";
  if (op.includes("point_mut")) return "point mutation";
  if (op.includes("reproduction")) return "reproduction";
  return op;
}

function fitnessOrLow(value: number | null): number {
  return value ?? Number.NEGATIVE_INFINITY;
}

function bestOf(values: Array<number | null>): number | null {
  const present = values.filter((v): v is number => v !== null && v !== undefined);
  return present.length ? Math.max(...present) : null;
}

export function groupLineage(lineage: Lineage): GenerationGroup[] {
  const byGen = new Map<number, LineageNode[]>();
  for (const node of lineage.nodes) {
    const bucket = byGen.get(node.generation) ?? [];
    bucket.push(node);
    byGen.set(node.generation, bucket);
  }

  const generations: GenerationGroup[] = [];
  for (const [generation, nodes] of byGen) {
    const byMethod = new Map<string, GroupMember[]>();
    for (const node of nodes) {
      const method = primaryMethod(node.op);
      const members = byMethod.get(method) ?? [];
      members.push({ id: node.id, fitness: node.fitness ?? null, op: node.op });
      byMethod.set(method, members);
    }

    const groups: MethodGroup[] = [];
    for (const [method, members] of byMethod) {
      members.sort((a, b) => fitnessOrLow(b.fitness) - fitnessOrLow(a.fitness));
      groups.push({
        method,
        members,
        count: members.length,
        bestFitness: bestOf(members.map((m) => m.fitness)),
      });
    }
    // most promising method first
    groups.sort((a, b) => fitnessOrLow(b.bestFitness) - fitnessOrLow(a.bestFitness));

    generations.push({
      generation,
      bestFitness: bestOf(nodes.map((n) => n.fitness ?? null)),
      groups,
    });
  }

  // latest generation on top
  generations.sort((a, b) => b.generation - a.generation);
  return generations;
}
