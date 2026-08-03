"""P9 - LLM credential resolution, mirroring the Tiingo key contract exactly.

Resolution order is env var > stored settings > none, per provider. The settings file may hold a
key, but ``GET /settings`` reports only *whether* one is set and *where it came from* — the
secret itself is never echoed, masked, sized, or otherwise leaked back to a caller.

Stored under ``meta/settings.json``:

    {"llm": {"provider": "openai", "model": "gpt-4o", "base_url": "",
             "keys": {"openai": "sk-..."}}}
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from alphalineage.data import paths
from alphalineage.explain.providers import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    PROVIDERS_BY_ID,
    ProviderSpec,
    get_provider,
)

#: Applies to any provider that has no key of its own; lets one env var drive a custom endpoint.
GENERIC_ENV_VAR = "ALPHALINEAGE_LLM_API_KEY"


@dataclass(frozen=True)
class ResolvedKey:
    """A key and where it came from. ``value`` never crosses the API boundary."""

    value: str
    source: str  # "environment" | "stored" | "none"

    @property
    def is_set(self) -> bool:
        return bool(self.value)


def _llm_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    stored = settings if settings is not None else paths.read_settings()
    section = stored.get("llm")
    return dict(section) if isinstance(section, dict) else {}


def _stored_keys(settings: dict[str, Any] | None = None) -> dict[str, str]:
    keys = _llm_settings(settings).get("keys")
    if not isinstance(keys, dict):
        return {}
    return {str(k): str(v) for k, v in keys.items() if isinstance(v, str) and v.strip()}


def resolve_key(provider_id: str, settings: dict[str, Any] | None = None) -> ResolvedKey:
    """Resolve a provider's key: env var first, then the stored settings, else none."""
    spec = PROVIDERS_BY_ID.get((provider_id or "").strip().lower())
    env_vars = spec.env_vars if spec else ()
    for name in (*env_vars, GENERIC_ENV_VAR):
        value = os.environ.get(name, "").strip()
        if value:
            return ResolvedKey(value, "environment")
    stored = _stored_keys(settings).get((provider_id or "").strip().lower(), "").strip()
    if stored:
        return ResolvedKey(stored, "stored")
    return ResolvedKey("", "none")


@dataclass(frozen=True)
class LLMConfig:
    """The effective LLM configuration: provider, model, endpoint, and key availability."""

    provider: str
    model: str
    base_url: str
    key: ResolvedKey

    @property
    def configured(self) -> bool:
        return self.key.is_set and bool(self.model)

    def public_dict(self) -> dict[str, Any]:
        """Safe to return over the API: says whether a key exists, never what it is."""
        return {
            "llm_provider": self.provider,
            "llm_model": self.model,
            "llm_base_url": self.base_url,
            "llm_api_key_set": self.key.is_set,
            "llm_api_key_source": self.key.source,
            "llm_configured": self.configured,
        }


def load_config(
    *,
    provider: str = "",
    model: str = "",
    base_url: str = "",
    settings: dict[str, Any] | None = None,
) -> LLMConfig:
    """Merge per-request overrides over the stored defaults, then resolve the key.

    A request may name a different provider or model than the saved default (the UI picker does
    exactly this) without changing what is persisted.
    """
    stored = _llm_settings(settings)
    chosen = (provider or str(stored.get("provider") or "") or DEFAULT_PROVIDER).strip().lower()
    spec = get_provider(chosen)
    stored_model = str(stored.get("model") or "") if chosen == stored.get("provider") else ""
    stored_base = str(stored.get("base_url") or "") if chosen == stored.get("provider") else ""
    return LLMConfig(
        provider=spec.id,
        model=(model or stored_model or spec.default_model).strip(),
        base_url=(base_url or stored_base or "").strip(),
        key=resolve_key(spec.id, settings),
    )


def provider_status(settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Catalog entries annotated with key availability, for the provider picker."""
    stored = settings if settings is not None else paths.read_settings()
    out: list[dict[str, Any]] = []
    for spec in PROVIDERS:
        key = resolve_key(spec.id, stored)
        out.append({**spec.to_dict(), "api_key_set": key.is_set, "api_key_source": key.source})
    return out


def update_settings(
    settings: dict[str, Any],
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_provider: str | None = None,
) -> dict[str, Any]:
    """Apply an LLM settings patch in place and return the mutated settings dict.

    An empty-string ``api_key`` clears the stored key for that provider, matching the Tiingo
    contract so the UI has one mental model for secrets.
    """
    section = _llm_settings(settings)
    if provider is not None:
        section["provider"] = get_provider(provider).id
    if model is not None:
        section["model"] = model.strip()
    if base_url is not None:
        section["base_url"] = base_url.strip()
    if api_key is not None:
        target = api_key_provider or provider or section.get("provider") or DEFAULT_PROVIDER
        spec: ProviderSpec = get_provider(str(target))
        keys = dict(_stored_keys(settings))
        if api_key.strip():
            keys[spec.id] = api_key.strip()
        else:
            keys.pop(spec.id, None)
        section["keys"] = keys
    settings["llm"] = section
    return settings
