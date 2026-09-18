"""P11 - the operator vocabulary the agent may use, and macro expansion for stored factors.

Both functions lived in ``explain/service.py`` while explanation was a feature of its own. That
module is gone; these two are the only parts worth keeping, and the agent is their only caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from alphalineage.core.tree import Node
from alphalineage.core.tree import from_dict as tree_from_dict
from alphalineage.explain.anatomy import base_name


def operator_listing(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The operators an expression may legally use.

    Restricted to the categories and formulas the session actually enabled, so the model cannot
    propose an operator the user's own search space excludes. Data fields and constants stay
    available regardless of the category gate — without them nothing is expressible at all.
    """
    from alphalineage.core.categories import builtin_category
    from alphalineage.core.primitive_docs import primitive_doc
    from alphalineage.core.primitives import REGISTRY
    from alphalineage.library.indicator_catalog import INDICATOR_CATALOG

    catalog_categories = {str(item["name"]): str(item["category"]) for item in INDICATOR_CATALOG}

    settings = config or {}
    enabled_categories = set(settings.get("enabled_categories") or [])
    enabled_formulas = set(settings.get("enabled_formula_names") or [])

    out: list[dict[str, Any]] = []
    for name, prim in sorted(REGISTRY.items()):
        is_macro = prim.macro_body is not None
        if is_macro:
            if enabled_formulas and base_name(name) not in enabled_formulas:
                continue
            category = catalog_categories.get(base_name(name), "technical_indicators")
        else:
            category = builtin_category(name)
            if enabled_categories and category not in enabled_categories:
                if category not in {"data", "constant"}:
                    continue
        doc = primitive_doc(name, prim.arity)
        out.append(
            {
                "name": name,
                "arg_types": [t.value for t in prim.arg_types],
                "out_type": prim.out_type.value,
                "category": category,
                "description": str(doc.get("description") or ""),
                "user": is_macro,
            }
        )
    return out


def expand_tree(
    tree: Node,
    stored_expanded: Any = None,
    *,
    pinned_specs: Sequence[dict[str, Any]] | None = None,
) -> Node:
    """Inline macros, preferring sources that do not depend on live registry state.

    Order matters. A stored expansion is exact. Failing that, a session's pinned
    ``formula_revisions`` describe every macro body the run actually used, which is what keeps an
    old session's factor measurable after its formulas have been edited or retired. The live
    registry is the last resort, and an unregistered macro degrades to surface-level analysis
    rather than failing the request.
    """
    if isinstance(stored_expanded, dict):
        try:
            return tree_from_dict(stored_expanded)
        except (KeyError, TypeError):
            pass
    if pinned_specs:
        try:
            from alphalineage.library.factors import expanded_snapshot

            specs = [
                {"name": str(spec.get("runtime_name") or spec.get("name")), "body": spec["body"]}
                for spec in pinned_specs
                if isinstance(spec, dict) and isinstance(spec.get("body"), dict)
            ]
            if specs:
                return expanded_snapshot(tree, specs)
        except Exception:  # noqa: BLE001 - a stale pin must not block measurement
            pass
    try:
        from alphalineage.core.extensions import expand_all

        return expand_all(tree)
    except Exception:  # noqa: BLE001 - an unregistered macro must not block measurement
        return tree
