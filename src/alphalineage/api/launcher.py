"""Local app entrypoint with process ownership and durable diagnostics on the project drive."""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from alphalineage.data.identifiers import atomic_write_text


def _creation_filetime() -> str | None:
    if os.name != "nt":
        return None
    from ctypes import wintypes

    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    times = [wintypes.FILETIME() for _ in range(4)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    if kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(t) for t in times)):
        return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
    return None


def record_shutdown(reason: str, exit_code: int) -> None:
    """Keep diagnostics even if the PowerShell task launcher has already terminated."""
    state_path = os.environ.get("ALPHALINEAGE_SERVER_STATE")
    if not state_path:
        return
    try:
        path = Path(state_path)
        state = json.loads(path.read_text(encoding="utf-8-sig"))
        if state.get("backend_pid") != os.getpid() or state.get("launch_id") != (
            os.environ.get("ALPHALINEAGE_INSTANCE_ID")
        ):
            return
        state.update(status="stopped", stopped_at=datetime.now(UTC).isoformat(),
                     exit_code=exit_code, shutdown_reason=reason)
        atomic_write_text(path, json.dumps(state, indent=2))
    except (OSError, ValueError, TypeError):
        logging.getLogger("uvicorn.error").exception("Cannot record backend shutdown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    runtime = Path(os.environ.get("ALPHALINEAGE_RUNTIME_DIR", Path.cwd() / ".runtime"))
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["ALPHALINEAGE_SERVER_STATE"] = str(runtime / f"server-{args.port}.json")
    state = {
        "launch_id": os.environ.get("ALPHALINEAGE_INSTANCE_ID"),
        "mode": os.environ.get("ALPHALINEAGE_LAUNCH_MODE", "foreground"),
        "launcher_pid": int(os.environ.get("ALPHALINEAGE_LAUNCHER_PID", os.getppid())),
        "backend_pid": os.getpid(),
        "backend_creation_filetime": _creation_filetime(),
        "started_at": datetime.now(UTC).isoformat(),
        "port": args.port,
        "status": "starting",
        "data_root": os.environ.get("ALPHALINEAGE_DATA_DIR"),
        "temp_root": os.environ.get("ALPHALINEAGE_TEMP_DIR"),
    }
    atomic_write_text(runtime / f"server-{args.port}.json", json.dumps(state, indent=2))
    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["file"] = {"format": "%(asctime)s %(levelname)s %(message)s"}
    log_config["handlers"]["backend_file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "formatter": "file",
        "filename": str(runtime / f"backend-{args.port}.log"),
        "encoding": "utf-8",
        "maxBytes": 20 * 1024**2,
        "backupCount": 2,
    }
    log_config["loggers"]["uvicorn"]["handlers"].append("backend_file")
    log_config["loggers"]["uvicorn.access"]["handlers"].append("backend_file")
    exit_code = 0
    try:
        uvicorn.run(
            "alphalineage.api.app:app", host="127.0.0.1", port=args.port,
            env_file=str(args.env_file) if args.env_file else None,
            log_config=log_config,
        )
    except Exception:
        exit_code = 1
        logging.getLogger("uvicorn.error").exception("Backend launcher failed")
        raise
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        raise
    finally:
        record_shutdown("server exited", exit_code)


if __name__ == "__main__":
    main()
