// B4 Design A (default): a collapsible per-generation list. Generations and method groups are
// collapsed by default and rendered only when expanded, so thousands of nodes stay legible and
// cheap. Expanding a group reveals its members in fitness-descending order.

import { useState } from "react";
import type { Lineage, LineageNode } from "../api/types";
import type { GenerationGroup } from "./groupLineage";

function fmt(value: number | null): string {
  return value === null ? "—" : value.toFixed(4);
}

export function GenerationList({
  generations,
  lineage,
  onSelect,
  onTrace,
  onSave,
}: {
  generations: GenerationGroup[];
  lineage: Lineage;
  onSelect: (id: number) => void;
  onTrace: (id: number) => void;
  onSave?: (node: LineageNode) => void;
}) {
  const [openGens, setOpenGens] = useState<Set<number>>(new Set());
  const [openGroups, setOpenGroups] = useState<Set<string>>(new Set());

  const toggle = <T,>(set: Set<T>, key: T): Set<T> => {
    const next = new Set(set);
    next.has(key) ? next.delete(key) : next.add(key);
    return next;
  };

  return (
    <div className="generation-list" data-testid="generation-list">
      {generations.map((gen) => {
        const genOpen = openGens.has(gen.generation);
        return (
          <div className="gen-block" key={gen.generation} data-testid="gen-block">
            <button
              type="button"
              className="gen-header"
              aria-expanded={genOpen}
              onClick={() => setOpenGens((s) => toggle(s, gen.generation))}
            >
              <span className="gen-caret">{genOpen ? "▼" : "▸"}</span>
              <span className="gen-title">Generation {gen.generation}</span>
              <span className="gen-best">best {fmt(gen.bestFitness)}</span>
            </button>

            {genOpen &&
              gen.groups.map((group) => {
                const key = `${gen.generation}:${group.method}`;
                const groupOpen = openGroups.has(key);
                return (
                  <div className="method-group" key={key} data-testid="method-group">
                    <button
                      type="button"
                      className="group-header"
                      aria-expanded={groupOpen}
                      onClick={() => setOpenGroups((s) => toggle(s, key))}
                    >
                      <span className="gen-caret">{groupOpen ? "▼" : "▸"}</span>
                      <span className="group-method">{group.method}</span>
                      <span className="group-count">({group.count})</span>
                      <span className="group-best">best {fmt(group.bestFitness)}</span>
                    </button>

                    {groupOpen && (
                      <ul className="member-list">
                        {group.members.map((member) => (
                          <li className="member-row" key={member.id} data-testid="member-row">
                            <button
                              type="button"
                              className="member-pick"
                              onClick={() => onSelect(member.id)}
                            >
                              #{member.id}
                            </button>
                            <span className="member-fitness">{fmt(member.fitness)}</span>
                            <span className="member-op">{member.op}</span>
                            {onSave && (
                              <button
                                type="button"
                                className="ghost"
                                data-testid="member-save"
                                onClick={() => {
                                  const node = lineage.nodes.find((n) => n.id === member.id);
                                  if (node) onSave(node);
                                }}
                              >
                                Save
                              </button>
                            )}
                            <button
                              type="button"
                              className="ghost"
                              data-testid="member-trace"
                              onClick={() => onTrace(member.id)}
                            >
                              Trace
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })}
          </div>
        );
      })}
    </div>
  );
}
