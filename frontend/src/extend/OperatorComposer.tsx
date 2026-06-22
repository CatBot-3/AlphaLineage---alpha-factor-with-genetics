// P7-T2: compose a custom operator as a typed macro using a React Flow editor.

import {
  addEdge,
  Background,
  type Connection,
  Controls,
  type Edge,
  type Node,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { addFormula, getPrimitives } from "../api/client";
import type { FactorNode, OperatorComposerDraft, PrimitiveInfo } from "../api/types";
import { type ComposerEdge, type ComposerNode, findRoot, graphToBody } from "./graphToBody";

const OUTPUT_TYPES = ["series", "signal", "window", "scalar"];

function nodesFromDraft(draft?: OperatorComposerDraft): Node[] {
  return (draft?.nodes ?? []).map((node) => ({
    id: node.id,
    position: { x: node.x, y: node.y },
    data: {
      label: node.label ?? node.kind,
      kind: node.kind,
      argIndex: node.argIndex,
      value: node.value,
    },
  }));
}

function edgesFromDraft(draft?: OperatorComposerDraft): Edge[] {
  return (draft?.edges ?? []).map((edge, i) => ({
    id: `${edge.source}-${edge.target}-${i}`,
    source: edge.source,
    target: edge.target,
  }));
}

export function OperatorComposer({
  draft,
  onDraftChange,
  canSubmit = true,
  onUseBody,
}: {
  draft?: OperatorComposerDraft;
  onDraftChange?: (draft: OperatorComposerDraft) => void;
  canSubmit?: boolean;
  onUseBody?: (body: FactorNode) => void;
}) {
  const initialNodes = useMemo(() => nodesFromDraft(draft), [draft]);
  const initialEdges = useMemo(() => edgesFromDraft(draft), [draft]);
  const [palette, setPalette] = useState<PrimitiveInfo[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initialEdges);
  const [name, setName] = useState(draft?.name ?? "my_op");
  const [argTypes, setArgTypes] = useState<string[]>(draft?.argTypes ?? ["series", "window"]);
  const [outType, setOutType] = useState(draft?.outType ?? "series");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const nextId = useRef(initialNodes.length);

  useEffect(() => {
    if (!canSubmit) {
      setPalette([]);
      return;
    }
    getPrimitives()
      .then(setPalette)
      .catch(() => setPalette([]));
  }, [canSubmit]);

  useEffect(() => {
    onDraftChange?.({
      name,
      argTypes,
      outType,
      nodes: nodes.map((node) => ({
        id: node.id,
        kind: String(node.data.kind),
        label: String(node.data.label ?? node.data.kind),
        argIndex: node.data.argIndex as number | undefined,
        value: node.data.value as number | undefined,
        x: node.position.x,
        y: node.position.y,
      })),
      edges: edges.map((edge) => ({ source: edge.source, target: edge.target })),
    });
  }, [argTypes, edges, name, nodes, onDraftChange, outType]);

  const onConnect = useCallback((c: Connection) => setEdges((es) => addEdge(c, es)), [setEdges]);

  function addNode(kind: string, extra: Record<string, unknown> = {}, label?: string) {
    const id = `c${nextId.current++}`;
    setNodes((ns) => [
      ...ns,
      {
        id,
        position: { x: 30 + ns.length * 40, y: 30 + ns.length * 55 },
        data: { label: label ?? kind, kind, ...extra },
      },
    ]);
  }

  function buildBody(): FactorNode | null {
    const composerNodes: ComposerNode[] = nodes.map((n) => ({
      id: n.id,
      kind: String(n.data.kind),
      argIndex: n.data.argIndex as number | undefined,
      value: n.data.value as number | undefined,
      x: n.position.x,
    }));
    const composerEdges: ComposerEdge[] = edges.map((e) => ({ source: e.source, target: e.target }));
    const root = findRoot(composerNodes, composerEdges);
    if (!root) {
      setError("the graph must have exactly one output (root) node");
      return null;
    }
    try {
      return graphToBody(composerNodes, composerEdges, root);
    } catch (e) {
      setError(String(e));
      return null;
    }
  }

  function useInEditor() {
    setError(null);
    const body = buildBody();
    if (body) onUseBody?.(body);
  }

  function register() {
    if (!canSubmit) return;
    setError(null);
    setMessage(null);
    const body = buildBody();
    if (!body) return;
    try {
      addFormula({
        name,
        display_name: name.replace(/_/g, " "),
        description: "Composed in the graph editor.",
        arg_types: argTypes,
        out_type: outType,
        body,
      })
        .then((p) =>
          setMessage(`Saved ${p.name}(${p.arg_types.join(", ")}) -> ${p.out_type}`),
        )
        .catch((e: Error) => setError(e.message));
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <section className="panel" data-testid="operator-composer">
      <h3>Compose an operator</h3>
      {!canSubmit && (
        <p className="panel-note">
          Static demo mode keeps this graph as a local draft; connect the backend to register it.
        </p>
      )}
      <div className="composer-signature">
        <label className="field">
          Name <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field">
          Args{" "}
          <input
            aria-label="arg-types"
            value={argTypes.join(",")}
            onChange={(e) =>
              setArgTypes(
                e.target.value
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean),
              )
            }
          />
        </label>
        <label className="field">
          Output{" "}
          <select value={outType} onChange={(e) => setOutType(e.target.value)}>
            {OUTPUT_TYPES.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </label>
        <button onClick={register} data-testid="register-operator" disabled={!canSubmit}>
          Register
        </button>
        {onUseBody && (
          <button type="button" onClick={useInEditor} data-testid="use-in-editor">
            Use in editor
          </button>
        )}
      </div>
      <div className="composer-palette" data-testid="palette">
        {argTypes.map((_, i) => (
          <button key={`arg${i}`} onClick={() => addNode("$arg", { argIndex: i }, `$arg ${i}`)}>
            $arg {i}
          </button>
        ))}
        <button onClick={() => addNode("const", { value: 1 }, "const 1")}>const</button>
        <button onClick={() => addNode("window", { value: 5 }, "window 5")}>window</button>
        {palette
          .filter((p) => p.kind !== "ephemeral")
          .map((p) => (
            <button
              key={p.name}
              className={p.user ? "palette-user" : undefined}
              onClick={() => addNode(p.name)}
            >
              {p.name}
            </button>
          ))}
      </div>
      <div className="graph">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>
      {message && (
        <p className="ok" data-testid="composer-result">
          {message}
        </p>
      )}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
