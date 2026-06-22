// Pure text <-> tree conversion for the formula editor (function-call syntax).
//
//   sub(ts_mean($0, 12), ts_mean($0, 26))
//
// `$N` is the N-th declared argument. A bare number is typed by its parent slot: a WINDOW
// slot becomes {name:"window", value}, a SCALAR slot becomes {name:"const", value}. Identifiers
// resolve to a primitive or user formula; a bare identifier with no parens is a leaf operand.
// The tree is the single source of truth; this module keeps the text pane in sync with it.

import type { FactorNode, PrimitiveInfo } from "../api/types";

export interface ParseError {
  pos: number;
  msg: string;
}

export interface ParseResult {
  tree?: FactorNode;
  errors: ParseError[];
}

type TokKind = "arg" | "number" | "ident" | "lparen" | "rparen" | "comma";
interface Token {
  kind: TokKind;
  text: string;
  pos: number;
}

const TOKEN_RE = /\s+|(\$\d+)|(-?\d+(?:\.\d+)?)|([a-zA-Z_][a-zA-Z0-9_]*)|(\()|(\))|(,)/y;

function tokenize(text: string): { tokens: Token[]; errors: ParseError[] } {
  const tokens: Token[] = [];
  const errors: ParseError[] = [];
  let i = 0;
  while (i < text.length) {
    TOKEN_RE.lastIndex = i;
    const m = TOKEN_RE.exec(text);
    if (!m || m.index !== i) {
      errors.push({ pos: i, msg: `unexpected character ${JSON.stringify(text[i])}` });
      i += 1;
      continue;
    }
    const [whole, arg, num, ident, lp, rp, comma] = m;
    i += whole.length;
    if (arg) tokens.push({ kind: "arg", text: arg, pos: m.index });
    else if (num) tokens.push({ kind: "number", text: num, pos: m.index });
    else if (ident) tokens.push({ kind: "ident", text: ident, pos: m.index });
    else if (lp) tokens.push({ kind: "lparen", text: lp, pos: m.index });
    else if (rp) tokens.push({ kind: "rparen", text: rp, pos: m.index });
    else if (comma) tokens.push({ kind: "comma", text: comma, pos: m.index });
    // else: whitespace, skip
  }
  return { tokens, errors };
}

export function parseFormula(
  text: string,
  argTypes: string[],
  primitives: PrimitiveInfo[],
): ParseResult {
  const byName = new Map(primitives.map((p) => [p.name, p]));
  const { tokens, errors } = tokenize(text);
  if (errors.length) return { errors };
  let idx = 0;

  const peek = (): Token | undefined => tokens[idx];
  const fail = (pos: number, msg: string): never => {
    throw { pos, msg } as ParseError;
  };

  function numberLeaf(tok: Token, expected: string | null): FactorNode {
    if (expected === "window") {
      if (!/^-?\d+$/.test(tok.text)) fail(tok.pos, "window must be a whole number");
      return { name: "window", value: parseInt(tok.text, 10) };
    }
    if (expected === "scalar") return { name: "const", value: parseFloat(tok.text) };
    fail(tok.pos, expected ? `a number is not valid where a ${expected} is expected` : "unexpected number");
    return { name: "const", value: 0 }; // unreachable
  }

  function parseExpr(expected: string | null): FactorNode {
    const tok = peek();
    if (!tok) return fail(text.length, "unexpected end of formula");
    if (tok.kind === "arg") {
      idx += 1;
      const i = parseInt(tok.text.slice(1), 10);
      if (i < 0 || i >= argTypes.length) fail(tok.pos, `$${i} is out of range (${argTypes.length} args)`);
      return { name: "$arg", value: i };
    }
    if (tok.kind === "number") {
      idx += 1;
      return numberLeaf(tok, expected);
    }
    if (tok.kind === "ident") {
      idx += 1;
      const prim = byName.get(tok.text);
      if (!prim) fail(tok.pos, `unknown function or field ${JSON.stringify(tok.text)}`);
      const next = peek();
      if (next?.kind === "lparen") {
        idx += 1; // consume (
        const children: FactorNode[] = [];
        if (peek()?.kind !== "rparen") {
          for (;;) {
            const expectedArg = prim!.arg_types[children.length] ?? null;
            children.push(parseExpr(expectedArg));
            const sep = peek();
            if (sep?.kind === "comma") {
              idx += 1;
              continue;
            }
            break;
          }
        }
        const close = peek();
        if (close?.kind !== "rparen") fail(close?.pos ?? text.length, "expected )");
        idx += 1; // consume )
        if (children.length !== prim!.arg_types.length) {
          fail(tok.pos, `${tok.text} expects ${prim!.arg_types.length} args, got ${children.length}`);
        }
        return { name: tok.text, children };
      }
      // bare identifier: must be a 0-arg primitive (an operand)
      if (prim!.arg_types.length !== 0) fail(tok.pos, `${tok.text} needs arguments: ${tok.text}(...)`);
      return { name: tok.text };
    }
    return fail(tok.pos, `unexpected ${tok.text}`);
  }

  try {
    const tree = parseExpr(null);
    if (idx < tokens.length) {
      const extra = tokens[idx];
      return { errors: [{ pos: extra.pos, msg: `unexpected ${JSON.stringify(extra.text)}` }] };
    }
    return { tree, errors: [] };
  } catch (e) {
    return { errors: [e as ParseError] };
  }
}

export function serializeFormula(tree: FactorNode): string {
  if (tree.name === "$arg") return `$${tree.value ?? 0}`;
  if (tree.children && tree.children.length) {
    return `${tree.name}(${tree.children.map(serializeFormula).join(", ")})`;
  }
  if (tree.value !== undefined) return String(tree.value);
  return tree.name;
}
