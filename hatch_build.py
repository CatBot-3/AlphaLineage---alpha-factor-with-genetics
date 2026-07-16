"""Hatch wheel metadata hook for the optional precompiled native evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Mark wheels platform/ABI-specific only when a native artifact is present."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        package = Path(self.root) / "src" / "alphalineage"
        native = any(package.glob("_evaluator*.pyd")) or any(package.glob("_evaluator*.so"))
        if native:
            build_data["infer_tag"] = True
            build_data["pure_python"] = False
