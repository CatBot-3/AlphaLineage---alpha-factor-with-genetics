"""PP-T2 - serialize a finished run into the static `demo` snapshot.

Pulls the run from a saved workspace (``data_cache/workspaces/run-{id}.json``) or a session
(``data_cache/sessions/{id}/result.json``), validates it, and writes the demo JSON the
zero-backend `demo` frontend build renders.

    python scripts/export_demo.py --workspace run-ab12cd34 --out frontend/public/demo-run.json
    python scripts/export_demo.py --session  myrun-ab12cd34 --out frontend/public/demo-run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphalineage.data import paths  # noqa: E402
from alphalineage.library.export import build_demo, validate_demo  # noqa: E402


def _run_from_workspace(workspace_id: str) -> dict:
    path = paths.workspaces_dir() / f"{workspace_id}.json"
    if not path.exists():
        raise SystemExit(f"no workspace at {path}")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    run = snapshot.get("run")
    if not run:
        raise SystemExit(f"workspace {workspace_id} has no run")
    return run


def _run_from_session(session_id: str) -> dict:
    path = paths.sessions_dir() / session_id / "result.json"
    if not path.exists():
        raise SystemExit(f"no session result at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a finished run to demo JSON.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workspace", help="workspace id (e.g. run-ab12cd34)")
    source.add_argument("--session", help="session id")
    parser.add_argument(
        "--out",
        default="frontend/public/demo-run.json",
        help="output path (default: frontend/public/demo-run.json)",
    )
    args = parser.parse_args()

    run = _run_from_workspace(args.workspace) if args.workspace else _run_from_session(args.session)
    demo = build_demo(run)
    validate_demo(demo)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(demo, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(demo['lineage']['nodes'])} lineage nodes)")


if __name__ == "__main__":
    main()
