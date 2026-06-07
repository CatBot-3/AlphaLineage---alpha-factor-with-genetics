// P7-T2: compose a custom operator as a typed macro using a React Flow drag-to-edit editor.
// The graph is converted to a body tree (graphToBody) and submitted to POST /operators — the
// server validates it as data (no code). Connections are made by dragging between node handles.

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
import { useCallback, useEffect, useState } from "react";
import { getPrimitives, registerOperator } from "../api/client";
import type { PrimitiveInfo } from "../api/types";
import { type ComposerEdge, type ComposerNode, findRoot, graphToBody } from "./graphToBody";

const OUTPUT_TYPES = ["series", "signal", "window", "scalar"];
let nodeCounter = 0;

export function OperatorComposer() {
  const [palette, setPalette] = useState<PrimitiveInfo[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [name, setName] = useState("my_op");
  const [argTypes, setArgTypes] = useState<string[]>(["series", "window"]);
  const [outType, setOutType] = useState("series");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPrimitives()
      .then(setPalette)
      .catch(() => setPalette([]));
  }, []);

  const onConnect = useCallback((c: Connection) => setEdges((es) => addEdge(c, es)), [setEdges]);

  function addNode(kind: string, extra: Record<string, unknown> = {}, label?: string) {
    const id = `c${nodeCounter++}`;
    setNodes((ns) => [
      ...ns,
      {
        id,
        position: { x: 30 + ns.length * 40, y: 30 + ns.length * 55 },
        data: { label: label ?? kind, kind, ...extra },
      },
    ]);
  }

  function register() {
    setError(null);
    setMessage(null);
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
      return;
    }
    try {
      const body = graphToBody(composerNodes, composerEdges, root);
      registerOperator({ name, arg_types: argTypes, out_type: outType, body })
        .then((p) => setMessage(`Registered ${p.name}(${p.arg_types.join(", ")}) -> ${p.out_type}`))
        .catch((e: Error) => setError(e.message));
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <section className="panel" data-testid="operator-composer">
      <h3>Compose an operator (no code — only existing primitives)</h3>
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
        <button onClick={register} data-testid="register-operator">
          Register
        </button>
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
            <button key={p.name} onClick={() => addNode(p.name)}>
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
