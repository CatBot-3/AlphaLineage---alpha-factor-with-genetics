"""Universe folders: a presentation-only hierarchy over universe names.

Folders never change what a universe *is* (its memberships, fingerprint, or the sessions pinned
to it); they only decide where the UI shows it. Bundled classification subsets (S&P 500 sectors
and themes) declare a default folder path in the packaged manifest, so a fresh install opens
with 30+ subsets tucked away instead of one long flat list. Users may create, rename, nest,
and delete folders, including the built-in ones, and move any universe in or out.

State lives in ``meta/universe_folders.json``::

    {
      "schema_version": 1,
      "folders": {"<id>": {"name": str, "parent": str | null}},   # user folders + overrides
      "deleted_builtin": ["<id>", ...],
      "placements": {"<universe>": "<folder id>" | null}          # explicit user choices
    }

Deleting a folder moves its child folders and universes to the deleted folder's parent; it
never deletes a universe.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphalineage.data import paths
from alphalineage.data.identifiers import atomic_write_text
from alphalineage.data.universe import bundled_snapshot_specs

SCHEMA_VERSION = 1
MAX_FOLDER_NAME = 80
MAX_DEPTH = 8
_BUILTIN_PREFIX = "builtin-folder-"
_USER_ID = re.compile(r"^fld-[0-9a-f]{12}$")
_lock = threading.RLock()


class FolderError(ValueError):
    """Invalid folder operation; ``status`` is the HTTP status the API should return."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Folder:
    id: str
    name: str
    parent: str | None
    builtin: bool

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "parent": self.parent, "builtin": self.builtin}


def folders_path() -> Path:
    return paths.meta_dir() / "universe_folders.json"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "folder"


def _builtin_id(path: tuple[str, ...]) -> str:
    return _BUILTIN_PREFIX + "--".join(_slug(part) for part in path)


def _builtin_layout() -> tuple[dict[str, Folder], dict[str, str]]:
    """Default folders and default placements declared by the packaged manifest."""
    folders: dict[str, Folder] = {}
    placements: dict[str, str] = {}
    for spec in bundled_snapshot_specs():
        path = tuple(str(part) for part in spec.get("folder") or [])
        if not path:
            continue
        parent: str | None = None
        for depth in range(1, len(path) + 1):
            folder_id = _builtin_id(path[:depth])
            folders.setdefault(folder_id, Folder(folder_id, path[depth - 1], parent, True))
            parent = folder_id
        assert parent is not None
        placements[str(spec["id"])] = parent
    return folders, placements


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "folders": {},
        "deleted_builtin": [],
        "placements": {},
    }


def _read_state() -> dict[str, Any]:
    path = folders_path()
    if not path.exists():
        return _empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt presentation file must never block research; fall back to defaults.
        return _empty_state()
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return _empty_state()
    state = _empty_state()
    raw_folders = payload.get("folders")
    if isinstance(raw_folders, dict):
        for folder_id, item in raw_folders.items():
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                parent = item.get("parent")
                state["folders"][str(folder_id)] = {
                    "name": item["name"],
                    "parent": parent if isinstance(parent, str) else None,
                }
    deleted = payload.get("deleted_builtin")
    if isinstance(deleted, list):
        state["deleted_builtin"] = [str(item) for item in deleted]
    placements = payload.get("placements")
    if isinstance(placements, dict):
        state["placements"] = {
            str(name): (value if isinstance(value, str) else None)
            for name, value in placements.items()
        }
    return state


def _write_state(state: dict[str, Any]) -> None:
    paths.meta_dir().mkdir(parents=True, exist_ok=True)
    atomic_write_text(folders_path(), json.dumps(state, indent=2, sort_keys=True))


def _effective_folders(state: dict[str, Any]) -> dict[str, Folder]:
    builtin, _ = _builtin_layout()
    deleted = set(state["deleted_builtin"])
    raw: dict[str, Folder] = {}
    for folder_id, folder in builtin.items():
        override = state["folders"].get(folder_id, {})
        raw[folder_id] = Folder(
            folder_id,
            str(override.get("name") or folder.name),
            override["parent"] if "parent" in override else folder.parent,
            True,
        )
    for folder_id, item in state["folders"].items():
        if folder_id in builtin or not _USER_ID.match(folder_id):
            continue
        raw[folder_id] = Folder(folder_id, item["name"], item.get("parent"), False)

    def live_parent(parent: str | None, seen: set[str]) -> str | None:
        # Walk past deleted or missing ancestors; a corrupt cycle degrades to the top level.
        while parent is not None and (parent in deleted or parent not in raw):
            if parent in seen or parent not in builtin:
                return None
            seen.add(parent)
            parent = builtin[parent].parent
        return parent

    live = {
        folder_id: Folder(folder.id, folder.name, live_parent(folder.parent, set()), folder.builtin)
        for folder_id, folder in raw.items()
        if folder_id not in deleted
    }
    # Break any cycle a hand-edited file might contain by lifting the offender to the top.
    fixed: dict[str, Folder] = {}
    for folder_id, folder in live.items():
        seen = {folder_id}
        parent = folder.parent
        cyclic = False
        while parent is not None:
            if parent in seen:
                cyclic = True
                break
            seen.add(parent)
            parent = live[parent].parent if parent in live else None
        fixed[folder_id] = (
            Folder(folder.id, folder.name, None, folder.builtin) if cyclic else folder
        )
    return fixed


def _resolve_placement(
    name: str,
    state: dict[str, Any],
    folders: dict[str, Folder],
    default_placements: dict[str, str],
) -> str | None:
    builtin, _ = _builtin_layout()
    if name in state["placements"]:
        target = state["placements"][name]
    else:
        target = default_placements.get(name)
    # A placement inside a deleted built-in folder follows that folder's contents upward.
    seen: set[str] = set()
    while target is not None and target not in folders:
        if target in seen or target not in builtin:
            return None
        seen.add(target)
        override = state["folders"].get(target, {})
        target = override["parent"] if "parent" in override else builtin[target].parent
    return target


def tree(universe_names: Iterable[str]) -> dict[str, Any]:
    """Folders plus the resolved folder (or ``None`` for top level) of every named universe."""
    with _lock:
        state = _read_state()
        folders = _effective_folders(state)
        _, defaults = _builtin_layout()
        return {
            "folders": [
                folder.as_dict()
                for folder in sorted(folders.values(), key=lambda item: (not item.builtin, item.id))
            ],
            "placements": {
                name: _resolve_placement(name, state, folders, defaults)
                for name in dict.fromkeys(universe_names)
            },
        }


def folder_path(folders: Iterable[dict[str, Any]], folder_id: str | None) -> list[str]:
    """Folder names from the top level down to ``folder_id`` (empty for the top level)."""
    by_id = {str(item["id"]): item for item in folders}
    names: list[str] = []
    seen: set[str] = set()
    while folder_id is not None and folder_id in by_id and folder_id not in seen:
        seen.add(folder_id)
        names.append(str(by_id[folder_id]["name"]))
        folder_id = by_id[folder_id]["parent"]
    return list(reversed(names))


def _clean_name(name: object) -> str:
    if not isinstance(name, str):
        raise FolderError("folder name must be text")
    clean = " ".join(name.split())
    if not clean:
        raise FolderError("folder name cannot be empty")
    if len(clean) > MAX_FOLDER_NAME:
        raise FolderError(f"folder name must be at most {MAX_FOLDER_NAME} characters")
    return clean


def _depth(folders: dict[str, Folder], folder_id: str | None) -> int:
    depth = 0
    while folder_id is not None:
        depth += 1
        folder_id = folders[folder_id].parent
    return depth


def _check_sibling_name(
    folders: dict[str, Folder], name: str, parent: str | None, exclude: str | None = None
) -> None:
    wanted = name.casefold()
    for folder in folders.values():
        if folder.id != exclude and folder.parent == parent and folder.name.casefold() == wanted:
            raise FolderError(f"a folder named {name!r} already exists here", status=409)


def _require_folder(folders: dict[str, Folder], folder_id: str | None) -> None:
    if folder_id is not None and folder_id not in folders:
        raise FolderError(f"unknown folder {folder_id!r}", status=404)


def create_folder(name: object, parent: str | None = None) -> dict[str, Any]:
    with _lock:
        state = _read_state()
        folders = _effective_folders(state)
        clean = _clean_name(name)
        _require_folder(folders, parent)
        if _depth(folders, parent) >= MAX_DEPTH:
            raise FolderError(f"folders can be nested at most {MAX_DEPTH} levels deep")
        _check_sibling_name(folders, clean, parent)
        folder = Folder(f"fld-{uuid.uuid4().hex[:12]}", clean, parent, False)
        state["folders"][folder.id] = {"name": folder.name, "parent": folder.parent}
        _write_state(state)
        return folder.as_dict()


_UNSET: Any = object()


def update_folder(folder_id: str, *, name: object = _UNSET, parent: Any = _UNSET) -> dict[str, Any]:
    with _lock:
        state = _read_state()
        folders = _effective_folders(state)
        _require_folder(folders, folder_id)
        current = folders[folder_id]
        new_name = current.name if name is _UNSET else _clean_name(name)
        new_parent = current.parent if parent is _UNSET else parent
        if new_parent is not None and not isinstance(new_parent, str):
            raise FolderError("parent must be a folder id or null")
        _require_folder(folders, new_parent)
        cursor = new_parent
        while cursor is not None:
            if cursor == folder_id:
                raise FolderError("a folder cannot be moved inside itself")
            cursor = folders[cursor].parent
        subtree_depth = _subtree_height(folders, folder_id)
        if _depth(folders, new_parent) + subtree_depth > MAX_DEPTH:
            raise FolderError(f"folders can be nested at most {MAX_DEPTH} levels deep")
        _check_sibling_name(folders, new_name, new_parent, exclude=folder_id)
        state["folders"][folder_id] = {"name": new_name, "parent": new_parent}
        _write_state(state)
        return Folder(folder_id, new_name, new_parent, current.builtin).as_dict()


def _subtree_height(folders: dict[str, Folder], folder_id: str) -> int:
    children = [folder.id for folder in folders.values() if folder.parent == folder_id]
    return 1 + max((_subtree_height(folders, child) for child in children), default=0)


def delete_folder(folder_id: str, universe_names: Iterable[str]) -> dict[str, Any]:
    """Remove a folder; its child folders and universes move to its parent."""
    with _lock:
        state = _read_state()
        folders = _effective_folders(state)
        _require_folder(folders, folder_id)
        target = folders[folder_id]
        _, defaults = _builtin_layout()
        # Pin the current location of everything that lives directly in the folder first, so the
        # move is explicit and survives future manifest changes.
        for name in dict.fromkeys(universe_names):
            if _resolve_placement(name, state, folders, defaults) == folder_id:
                state["placements"][name] = target.parent
        for folder in folders.values():
            if folder.parent == folder_id:
                state["folders"][folder.id] = {"name": folder.name, "parent": target.parent}
        if target.builtin:
            state["folders"].pop(folder_id, None)
            if folder_id not in state["deleted_builtin"]:
                state["deleted_builtin"].append(folder_id)
        else:
            state["folders"].pop(folder_id, None)
        _write_state(state)
        return {"removed": folder_id, "moved_to": target.parent}


def move_universes(names: Iterable[str], folder_id: str | None) -> dict[str, Any]:
    with _lock:
        state = _read_state()
        folders = _effective_folders(state)
        _require_folder(folders, folder_id)
        moved = list(dict.fromkeys(names))
        for name in moved:
            state["placements"][name] = folder_id
        _write_state(state)
        return {"universes": moved, "folder": folder_id}


def forget_universe(name: str) -> None:
    """Drop an explicit placement when its universe is deleted."""
    with _lock:
        state = _read_state()
        if name in state["placements"]:
            del state["placements"][name]
            _write_state(state)
