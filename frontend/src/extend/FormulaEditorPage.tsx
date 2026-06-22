// Extend > Formula Editor: author functions by text (kept in sync with a node display) or by
// dragging in the graph composer, customize every parameter, promote parameters to GP-tunable
// arguments, branch from any function, and organize functions into categories.

import { useEffect, useMemo, useRef, useState } from "react";
import {
  addFormula,
  deleteFormula,
  getCategories,
  getPrimitives,
  listFormulas,
  putCategories,
  setPrimitiveCategory,
  updateFormula,
  validateFormula,
} from "../api/client";
import type {
  CategorySettings,
  FactorNode,
  FormulaSpec,
  FormulaValidation,
  OperatorComposerDraft,
  PrimitiveInfo,
} from "../api/types";
import { OperatorComposer } from "./OperatorComposer";
import { parseFormula, serializeFormula } from "./formulaText";
import { leafType, paramLeaves, promoteToArg, setLeafValue } from "./treeEdit";

const TYPE_OPTIONS = ["series", "signal", "window", "scalar", "bool"];
const DEFAULT_TREE: FactorNode = { name: "rank", children: [{ name: "$arg", value: 0 }] };

function skeleton(prim: PrimitiveInfo): string {
  if (prim.arg_types.length === 0) return prim.name;
  const args = prim.arg_types.map((t) => (t === "window" ? "10" : t === "scalar" ? "1" : "$0"));
  return `${prim.name}(${args.join(", ")})`;
}

function NodeView({ node, userNames }: { node: FactorNode; userNames: Set<string> }) {
  if (node.name === "$arg") return <span className="fnode fnode-arg">${node.value ?? 0}</span>;
  if (!node.children?.length && node.value !== undefined) {
    return <span className="fnode fnode-leaf">{node.value}</span>;
  }
  if (!node.children?.length) return <span className="fnode fnode-operand">{node.name}</span>;
  const isUser = userNames.has(node.name);
  return (
    <span className={`fnode fnode-op${isUser ? " fnode-user" : ""}`}>
      <span className="fnode-name">{node.name}</span>
      <span className="fnode-args">
        {node.children.map((c, i) => (
          <NodeView key={i} node={c} userNames={userNames} />
        ))}
      </span>
    </span>
  );
}

export function FormulaEditorPage({
  operatorDraft,
  onOperatorDraftChange,
  canSubmit = true,
}: {
  operatorDraft?: OperatorComposerDraft;
  onOperatorDraftChange?: (draft: OperatorComposerDraft) => void;
  canSubmit?: boolean;
}) {
  const [primitives, setPrimitives] = useState<PrimitiveInfo[]>([]);
  const [formulas, setFormulas] = useState<FormulaSpec[]>([]);
  const [categories, setCategories] = useState<CategorySettings>({ order: [], overrides: {} });

  const [name, setName] = useState("my_formula");
  const [argTypes, setArgTypes] = useState<string[]>(["series"]);
  const [outType, setOutType] = useState("signal");
  const [category, setCategory] = useState("custom");
  const [description, setDescription] = useState("");
  const [tree, setTree] = useState<FactorNode>(DEFAULT_TREE);
  const [text, setText] = useState(serializeFormula(DEFAULT_TREE));
  const [loadedName, setLoadedName] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [validation, setValidation] = useState<FormulaValidation | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newCategory, setNewCategory] = useState("");
  const textRef = useRef<HTMLTextAreaElement>(null);

  const userNames = useMemo(() => new Set(formulas.map((f) => f.name)), [formulas]);
  const leaves = useMemo(() => paramLeaves(tree), [tree]);
  // Editable numeric leaves (window/const) get contiguous testids; $arg leaves are read-only.
  const numericLeaves = useMemo(() => leaves.filter((l) => l.node.name !== "$arg"), [leaves]);
  const argLeaves = useMemo(() => leaves.filter((l) => l.node.name === "$arg"), [leaves]);

  function refreshLists() {
    if (!canSubmit) return;
    getPrimitives().then(setPrimitives).catch(() => setPrimitives([]));
    listFormulas().then(setFormulas).catch(() => setFormulas([]));
    getCategories().then(setCategories).catch(() => undefined);
  }

  useEffect(refreshLists, [canSubmit]);

  // Programmatic tree edits update text too; raw text edits only update the tree (preserve caret).
  function setTreeAndText(next: FactorNode) {
    setTree(next);
    setText(serializeFormula(next));
  }

  function onTextChange(value: string) {
    setText(value);
    const { tree: parsed, errors } = parseFormula(value, argTypes, primitives);
    if (parsed) {
      setTree(parsed);
      setParseError(null);
    } else {
      setParseError(errors[0] ? `${errors[0].msg} (at ${errors[0].pos})` : "invalid formula");
    }
  }

  async function runValidate() {
    if (!canSubmit) return;
    try {
      setValidation(await validateFormula(buildSpec()));
    } catch (e) {
      setError(String(e));
    }
  }

  function buildSpec(): FormulaSpec {
    return {
      name,
      display_name: name.replace(/_/g, " "),
      description,
      arg_types: argTypes,
      out_type: outType,
      body: tree,
      category,
    };
  }

  function insertSkeleton(prim: PrimitiveInfo) {
    const snippet = skeleton(prim);
    const el = textRef.current;
    if (el && typeof el.selectionStart === "number") {
      const start = el.selectionStart;
      const end = el.selectionEnd;
      onTextChange(text.slice(0, start) + snippet + text.slice(end));
    } else {
      onTextChange(text ? `${text} ${snippet}` : snippet);
    }
  }

  function loadFormula(spec: FormulaSpec) {
    setName(spec.name);
    setArgTypes(spec.arg_types);
    setOutType(spec.out_type);
    setCategory(spec.category || "custom");
    setDescription(spec.description ?? "");
    setTreeAndText(spec.body);
    setLoadedName(spec.name);
    setMessage(null);
    setError(null);
    setValidation(null);
  }

  function branch() {
    setName(`${name}_copy`);
    setLoadedName(null); // a branch saves as a new formula, leaving the original intact
    setMessage("Branched - editing a new copy; Save to keep it.");
  }

  async function save() {
    if (!canSubmit) return;
    setError(null);
    setMessage(null);
    try {
      const spec = buildSpec();
      const saved = loadedName && loadedName === name ? await updateFormula(name, spec) : await addFormula(spec);
      setLoadedName(saved.name);
      setMessage(`Saved ${saved.name}`);
      refreshLists();
    } catch (e) {
      setError(String(e));
    }
  }

  async function remove(target: string) {
    if (!canSubmit) return;
    try {
      await deleteFormula(target);
      if (loadedName === target) setLoadedName(null);
      refreshLists();
    } catch (e) {
      setError(String(e));
    }
  }

  function setLeaf(path: number[], value: number) {
    setTreeAndText(setLeafValue(tree, path, value));
  }

  function promote(path: number[], type: string) {
    const nextArgs = [...argTypes, type];
    setArgTypes(nextArgs);
    setTreeAndText(promoteToArg(tree, path, nextArgs.length - 1));
    setMessage("Promoted to argument - the GP can now tune this value during training.");
  }

  async function recategorize(prim: string, value: string) {
    if (!canSubmit) return;
    setCategories(await setPrimitiveCategory(prim, value));
    refreshLists();
  }

  async function addCategory() {
    const trimmed = newCategory.trim();
    if (!trimmed || !canSubmit) return;
    setCategories(await putCategories({ order: [...categories.order, trimmed] }));
    setNewCategory("");
  }

  const grouped = useMemo(() => {
    const groups = new Map<string, PrimitiveInfo[]>();
    for (const p of primitives) {
      if (p.kind === "ephemeral") continue;
      const key = p.category ?? "uncategorized";
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(p);
    }
    return groups;
  }, [primitives]);

  return (
    <div className="formula-editor-page" data-testid="formula-editor-page">
      <section className="panel">
        <header className="panel-head">
          <div>
            <h3>Formula editor</h3>
            <p className="panel-note">
              Compose functions from existing ones; the text and the node view stay in sync.
              Promote a constant to an argument to let training tune it.
            </p>
          </div>
          <span className="mode-chip">{canSubmit ? `${formulas.length} saved` : "Local draft"}</span>
        </header>

        {!canSubmit && (
          <p className="panel-note">Saving and validation unlock when the backend is running.</p>
        )}

        <div className="formula-signature">
          <label className="field">
            <span className="field-label">Name</span>
            <input aria-label="Formula name" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">Arguments (comma types)</span>
            <input
              aria-label="Formula arg types"
              value={argTypes.join(", ")}
              onChange={(e) =>
                setArgTypes(e.target.value.split(",").map((s) => s.trim()).filter(Boolean))
              }
            />
          </label>
          <label className="field">
            <span className="field-label">Output</span>
            <select aria-label="Formula output type" value={outType} onChange={(e) => setOutType(e.target.value)}>
              {TYPE_OPTIONS.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Category</span>
            <select aria-label="Formula category" value={category} onChange={(e) => setCategory(e.target.value)}>
              {[...new Set([category, ...categories.order, "custom"])].map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
        </div>

        <label className="field formula-text-field">
          <span className="field-label">Formula text</span>
          <textarea
            ref={textRef}
            aria-label="Formula text"
            className="formula-text"
            rows={3}
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            onBlur={runValidate}
          />
        </label>
        {parseError && <p className="error">{parseError}</p>}
        {validation && (
          <p className={validation.ok ? "ok" : "error"} data-testid="formula-validation">
            {validation.ok ? `Valid - produces ${validation.out_type}` : validation.error}
          </p>
        )}

        <div className="formula-tree" data-testid="formula-tree">
          <NodeView node={tree} userNames={userNames} />
        </div>

        {leaves.length > 0 && (
          <div className="param-pane" data-testid="param-pane">
            <span className="field-label">Parameters</span>
            {numericLeaves.map((leaf, i) => (
              <span key={`p${i}`} className="param-row">
                <code>{leaf.node.name}</code>
                <input
                  type="number"
                  aria-label={`param-${i}`}
                  value={leaf.node.value ?? 0}
                  onChange={(e) => setLeaf(leaf.path, Number(e.target.value))}
                />
                <button
                  type="button"
                  className="ghost"
                  data-testid={`promote-${i}`}
                  onClick={() => promote(leaf.path, leafType(leaf.node, argTypes))}
                >
                  Promote to argument
                </button>
              </span>
            ))}
            {argLeaves.map((leaf, i) => (
              <span key={`a${i}`} className="param-row">
                <code>${leaf.node.value ?? 0}</code> <em>(argument)</em>
              </span>
            ))}
          </div>
        )}

        <div className="actions">
          <button type="button" className="primary-action" data-testid="save-formula" onClick={save} disabled={!canSubmit}>
            {loadedName && loadedName === name ? "Update formula" : "Save formula"}
          </button>
          <button type="button" className="ghost" data-testid="branch-formula" onClick={branch}>
            Branch as new
          </button>
        </div>
        {message && <p className="ok">{message}</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="panel">
        <h3>Building blocks</h3>
        <p className="panel-note">Click a function to insert it; re-categorize any function below.</p>
        {[...grouped.entries()].map(([cat, prims]) => (
          <div key={cat} className="function-group" data-testid={`group-${cat}`}>
            <span className="field-label">{cat}</span>
            <div className="function-chips">
              {prims.map((p) => (
                <span key={p.name} className="function-chip">
                  <button
                    type="button"
                    className={p.user ? "palette-user" : undefined}
                    onClick={() => insertSkeleton(p)}
                  >
                    {p.name}
                  </button>
                  {canSubmit && (
                    <select
                      aria-label={`category-${p.name}`}
                      value={p.category ?? "uncategorized"}
                      onChange={(e) => recategorize(p.name, e.target.value)}
                    >
                      {[...new Set([p.category ?? "uncategorized", ...categories.order])].map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                  )}
                </span>
              ))}
            </div>
          </div>
        ))}
        <div className="inline-tools">
          <label className="field">
            <span className="field-label">New category</span>
            <input
              aria-label="New category"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
            />
          </label>
          <button type="button" className="ghost" onClick={addCategory} disabled={!canSubmit}>
            Add category
          </button>
        </div>
      </section>

      {formulas.length > 0 && (
        <section className="panel">
          <h3>Saved formulas</h3>
          <ul className="formula-list" data-testid="formula-list">
            {formulas.map((f) => (
              <li key={f.name}>
                <div>
                  <strong>{f.display_name || f.name}</strong>
                  <span>
                    {f.name}({f.arg_types.join(", ")}) {"->"} {f.out_type} · {f.category || "custom"}
                  </span>
                  {f.error && <span className="error">{f.error}</span>}
                </div>
                <button type="button" className="ghost" onClick={() => loadFormula(f)}>
                  Edit
                </button>
                <button type="button" className="ghost" onClick={() => remove(f.name)}>
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <OperatorComposer
        draft={operatorDraft}
        onDraftChange={onOperatorDraftChange}
        canSubmit={canSubmit}
        onUseBody={setTreeAndText}
      />
    </div>
  );
}
