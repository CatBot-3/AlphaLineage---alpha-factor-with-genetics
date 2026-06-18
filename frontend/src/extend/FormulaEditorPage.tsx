// Extend > Formula Editor: save quick formula templates and compose operators from existing
// primitives. Pure relocation of the original "Formula library" section + OperatorComposer;
// composing-from-existing-formulas is a deferred design (not implemented here yet).

import { useEffect, useState } from "react";
import { addFormula, deleteFormula, listFormulas } from "../api/client";
import type { FactorNode, FormulaDraft, FormulaSpec, OperatorComposerDraft } from "../api/types";
import { OperatorComposer } from "./OperatorComposer";

interface FormulaTemplate {
  id: string;
  label: string;
  name: string;
  display_name: string;
  description: string;
  arg_types: string[];
  out_type: string;
  body: FactorNode;
}

const arg = (index: number): FactorNode => ({ name: "$arg", value: index });
const windowNode = (value: number): FactorNode => ({ name: "window", value });
const op = (name: string, children: FactorNode[]): FactorNode => ({ name, children });

const FORMULA_TEMPLATES: FormulaTemplate[] = [
  {
    id: "moving_average",
    label: "Moving average",
    name: "moving_average",
    display_name: "Moving average",
    description: "Trailing mean for a series with a caller-provided window.",
    arg_types: ["series", "window"],
    out_type: "series",
    body: op("ts_mean", [arg(0), arg(1)]),
  },
  {
    id: "momentum",
    label: "Momentum",
    name: "momentum",
    display_name: "Momentum",
    description: "Current value minus a delayed value.",
    arg_types: ["series", "window"],
    out_type: "series",
    body: op("sub", [arg(0), op("delay", [arg(0), arg(1)])]),
  },
  {
    id: "macd_spread",
    label: "Macd-style spread",
    name: "macd_spread",
    display_name: "Macd-style spread",
    description: "Twelve-day mean minus twenty-six-day mean.",
    arg_types: ["series"],
    out_type: "series",
    body: op("sub", [
      op("ts_mean", [arg(0), windowNode(12)]),
      op("ts_mean", [arg(0), windowNode(26)]),
    ]),
  },
  {
    id: "bollinger_zscore",
    label: "Bollinger z-score",
    name: "bollinger_zscore",
    display_name: "Bollinger z-score",
    description: "Distance from a twenty-day mean, scaled by twenty-day volatility.",
    arg_types: ["series"],
    out_type: "series",
    body: op("div", [
      op("sub", [arg(0), op("ts_mean", [arg(0), windowNode(20)])]),
      op("ts_std", [arg(0), windowNode(20)]),
    ]),
  },
];

function formulaTemplate(id: string | undefined): FormulaTemplate {
  return FORMULA_TEMPLATES.find((template) => template.id === id) ?? FORMULA_TEMPLATES[0];
}

export function FormulaEditorPage({
  formulaDraft,
  onFormulaDraftChange,
  operatorDraft,
  onOperatorDraftChange,
  canSubmit = true,
}: {
  formulaDraft?: FormulaDraft;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
  operatorDraft?: OperatorComposerDraft;
  onOperatorDraftChange?: (draft: OperatorComposerDraft) => void;
  canSubmit?: boolean;
}) {
  const initialFormula = formulaTemplate(formulaDraft?.template);
  const [formulas, setFormulas] = useState<FormulaSpec[]>([]);
  const [formulaTemplateId, setFormulaTemplateId] = useState(initialFormula.id);
  const [formulaName, setFormulaName] = useState(formulaDraft?.name ?? initialFormula.name);
  const [formulaDisplay, setFormulaDisplay] = useState(
    formulaDraft?.display_name ?? initialFormula.display_name,
  );
  const [formulaDescription, setFormulaDescription] = useState(
    formulaDraft?.description ?? initialFormula.description,
  );
  const [formulaArgTypes, setFormulaArgTypes] = useState<string[]>(
    formulaDraft?.arg_types ?? initialFormula.arg_types,
  );
  const [formulaOutType, setFormulaOutType] = useState(
    formulaDraft?.out_type ?? initialFormula.out_type,
  );
  const [formulaBody, setFormulaBody] = useState<FactorNode>(
    formulaDraft?.body ?? initialFormula.body,
  );
  const [formulaMessage, setFormulaMessage] = useState<string | null>(null);
  const [formulaError, setFormulaError] = useState<string | null>(null);

  useEffect(() => {
    onFormulaDraftChange?.({
      name: formulaName,
      display_name: formulaDisplay,
      description: formulaDescription,
      template: formulaTemplateId,
      arg_types: formulaArgTypes,
      out_type: formulaOutType,
      body: formulaBody,
    });
  }, [
    formulaArgTypes,
    formulaBody,
    formulaDescription,
    formulaDisplay,
    formulaName,
    formulaOutType,
    formulaTemplateId,
    onFormulaDraftChange,
  ]);

  useEffect(() => {
    if (!canSubmit) {
      setFormulas([]);
      return;
    }
    let cancelled = false;
    listFormulas()
      .then((items) => {
        if (!cancelled) setFormulas(items);
      })
      .catch(() => {
        if (!cancelled) setFormulas([]);
      });
    return () => {
      cancelled = true;
    };
  }, [canSubmit]);

  function applyTemplate(templateId: string) {
    const template = formulaTemplate(templateId);
    setFormulaTemplateId(template.id);
    setFormulaName(template.name);
    setFormulaDisplay(template.display_name);
    setFormulaDescription(template.description);
    setFormulaArgTypes(template.arg_types);
    setFormulaOutType(template.out_type);
    setFormulaBody(template.body);
    setFormulaError(null);
    setFormulaMessage(null);
  }

  async function saveFormula() {
    if (!canSubmit) return;
    setFormulaError(null);
    setFormulaMessage(null);
    const spec: FormulaSpec = {
      name: formulaName,
      display_name: formulaDisplay,
      description: formulaDescription,
      arg_types: formulaArgTypes,
      out_type: formulaOutType,
      body: formulaBody,
    };
    try {
      const saved = await addFormula(spec);
      setFormulas((current) => [...current.filter((item) => item.name !== saved.name), saved]);
      setFormulaMessage(`Saved formula ${saved.display_name || saved.name}`);
    } catch (error) {
      setFormulaError(String(error));
    }
  }

  async function removeFormula(nameToRemove: string) {
    if (!canSubmit) return;
    setFormulaError(null);
    setFormulaMessage(null);
    try {
      await deleteFormula(nameToRemove);
      setFormulas((current) => current.filter((item) => item.name !== nameToRemove));
      setFormulaMessage(`Deleted formula ${nameToRemove}`);
    } catch (error) {
      setFormulaError(String(error));
    }
  }

  return (
    <div className="formula-editor-page" data-testid="formula-editor-page">
      <section className="panel">
        <header className="panel-head">
          <div>
            <h3>Formula library</h3>
            <p className="panel-note">
              Save reusable formulas from templates. Composing your own from existing formulas is
              coming later - for now, start from a template and adjust it below.
            </p>
          </div>
          <span className="mode-chip">
            {canSubmit ? `${formulas.length} saved formulas` : "Local draft"}
          </span>
        </header>

        {!canSubmit && (
          <p className="panel-note">
            Static demo mode keeps formula drafts locally. Save/delete unlock when the backend is
            running.
          </p>
        )}

        <div className="template-grid">
          {FORMULA_TEMPLATES.map((template) => (
            <button
              key={template.id}
              type="button"
              aria-pressed={formulaTemplateId === template.id}
              onClick={() => applyTemplate(template.id)}
            >
              <strong>{template.label}</strong>
              <span>{template.description}</span>
            </button>
          ))}
        </div>

        <div className="formula-editor">
          <label className="field">
            <span className="field-label">Internal name</span>
            <input
              aria-label="Formula name"
              value={formulaName}
              onChange={(event) => setFormulaName(event.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Display name</span>
            <input
              aria-label="Formula display name"
              value={formulaDisplay}
              onChange={(event) => setFormulaDisplay(event.target.value)}
            />
          </label>
          <label className="field formula-description">
            <span className="field-label">Description</span>
            <input
              aria-label="Formula description"
              value={formulaDescription}
              onChange={(event) => setFormulaDescription(event.target.value)}
            />
          </label>
          <div className="formula-signature">
            <span>Args: {formulaArgTypes.join(", ")}</span>
            <span>Output: {formulaOutType}</span>
            <span>Body root: {formulaBody.name}</span>
          </div>
        </div>

        <div className="actions">
          <button type="button" data-testid="save-formula" onClick={saveFormula} disabled={!canSubmit}>
            Save formula
          </button>
        </div>

        {formulas.length > 0 && (
          <ul className="formula-list" data-testid="formula-list">
            {formulas.map((formula) => (
              <li key={formula.name}>
                <div>
                  <strong>{formula.display_name || formula.name}</strong>
                  <span>
                    {formula.name}({formula.arg_types.join(", ")}) {"->"} {formula.out_type}
                  </span>
                  {formula.error && <span className="error">{formula.error}</span>}
                </div>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => removeFormula(formula.name)}
                  disabled={!canSubmit}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
        {formulaMessage && <p className="ok">{formulaMessage}</p>}
        {formulaError && <p className="error">{formulaError}</p>}
      </section>

      <OperatorComposer draft={operatorDraft} onDraftChange={onOperatorDraftChange} canSubmit={canSubmit} />
    </div>
  );
}
