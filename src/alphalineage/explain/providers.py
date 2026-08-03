"""P9 - LLM provider catalog and clients. The only outbound network call this app makes
besides market data.

Two wire formats cover the supported providers: OpenAI Chat Completions (OpenAI, DeepSeek, and
any OpenAI-compatible endpoint including local Ollama / LM Studio / vLLM) and Anthropic Messages.

Everything here is defensive by construction, because this is a user-supplied URL carrying a
user-supplied secret:

* the host must match the provider's allowlisted host — only the explicit ``openai_compatible``
  provider accepts an arbitrary base URL, and even then only over HTTPS or loopback;
* the key never appears in a log line, an error message, or a persisted artifact
  (:func:`redact` is applied on every path out);
* responses are size-capped and retried with bounded backoff on 429/5xx only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from alphalineage.data.retry import with_retry

#: Hard cap on a response body. A model cannot be trusted to be brief and memory is finite.
MAX_RESPONSE_BYTES = 4_000_000
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


class LLMError(RuntimeError):
    """Any failure talking to a provider. The message is always redacted."""


@dataclass(frozen=True)
class ProviderSpec:
    """A supported provider: how to reach it, how to authenticate, what to call it."""

    id: str
    label: str
    wire: str  # "openai" | "anthropic"
    default_base_url: str
    default_model: str
    models: tuple[str, ...]
    env_vars: tuple[str, ...]
    allowed_hosts: tuple[str, ...] = ()
    allows_custom_base_url: bool = False
    docs_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "wire": self.wire,
            "default_base_url": self.default_base_url,
            "default_model": self.default_model,
            "models": list(self.models),
            "env_vars": list(self.env_vars),
            "allows_custom_base_url": self.allows_custom_base_url,
            "docs_url": self.docs_url,
        }


#: Ordered by the user's stated priority: OpenAI, Anthropic, DeepSeek, then the escape hatch.
#: ``models`` are suggestions for the picker, not a whitelist — the model string is free text so
#: a newly released model works the day it ships without a code change.
PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openai",
        label="OpenAI",
        wire="openai",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        models=("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini"),
        env_vars=("OPENAI_API_KEY",),
        allowed_hosts=("api.openai.com",),
        docs_url="https://platform.openai.com/api-keys",
    ),
    ProviderSpec(
        id="anthropic",
        label="Anthropic",
        wire="anthropic",
        default_base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-5",
        models=(
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ),
        env_vars=("ANTHROPIC_API_KEY",),
        allowed_hosts=("api.anthropic.com",),
        docs_url="https://console.anthropic.com/settings/keys",
    ),
    ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        wire="openai",
        default_base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        env_vars=("DEEPSEEK_API_KEY",),
        allowed_hosts=("api.deepseek.com",),
        docs_url="https://platform.deepseek.com/api_keys",
    ),
    ProviderSpec(
        id="openai_compatible",
        label="OpenAI-compatible endpoint (custom)",
        wire="openai",
        default_base_url="",
        default_model="",
        models=(),
        env_vars=("ALPHALINEAGE_LLM_API_KEY",),
        allows_custom_base_url=True,
        docs_url="",
    ),
)

PROVIDERS_BY_ID: dict[str, ProviderSpec] = {spec.id: spec for spec in PROVIDERS}
DEFAULT_PROVIDER = "openai"


def get_provider(provider_id: str) -> ProviderSpec:
    spec = PROVIDERS_BY_ID.get((provider_id or "").strip().lower())
    if spec is None:
        raise LLMError(
            f"unknown provider {provider_id!r}; expected one of "
            f"{', '.join(sorted(PROVIDERS_BY_ID))}"
        )
    return spec


def redact(text: str, *secrets: str) -> str:
    """Replace every non-trivial secret with a placeholder. Applied to everything that escapes."""
    out = text
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***redacted***")
    return out


def resolve_base_url(spec: ProviderSpec, base_url: str = "") -> str:
    """Validate and return the endpoint root, enforcing the egress allowlist."""
    candidate = (base_url or spec.default_base_url or "").strip().rstrip("/")
    if not candidate:
        raise LLMError(f"provider {spec.id} requires a base URL")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LLMError("base URL must be an absolute http(s) URL")
    host = parsed.hostname.lower()
    loopback = host in _LOOPBACK_HOSTS
    if parsed.scheme != "https" and not loopback:
        raise LLMError("base URL must use https (plain http is allowed only for localhost)")
    if not spec.allows_custom_base_url:
        if host not in spec.allowed_hosts:
            raise LLMError(
                f"provider {spec.id} may only call {', '.join(spec.allowed_hosts)}; "
                f"use the 'openai_compatible' provider for a custom endpoint"
            )
    return candidate


@dataclass(frozen=True)
class Completion:
    """A provider's answer, normalized across wire formats."""

    text: str
    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str = ""
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_reason": self.stop_reason,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class LLMRequest:
    """Everything needed for one call. ``api_key`` is never persisted or echoed."""

    provider: str
    model: str
    api_key: str
    system: str
    user: str
    base_url: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout_s: float = DEFAULT_TIMEOUT_S
    extra_headers: dict[str, str] = field(default_factory=dict)


class Transport(Protocol):
    """The seam the tests inject through; production passes ``requests.post``."""

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> Any: ...


def _default_transport(
    url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
) -> Any:
    return requests.post(url, headers=headers, json=json, timeout=timeout)


def _retryable(status: int) -> bool:
    return status == 429 or 500 <= status < 600


class _RetryableStatus(Exception):
    def __init__(self, status: int, body: str, retry_after: float | None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body
        self.retry_after = retry_after


def _body_text(response: Any) -> str:
    try:
        text = response.text or ""
    except Exception:  # noqa: BLE001 - a mock or a broken stream must not mask the real error
        return ""
    return text[:2000]


def _retry_after(exc: BaseException) -> float | None:
    return exc.retry_after if isinstance(exc, _RetryableStatus) else None


def _post(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    transport: Transport,
    secrets: tuple[str, ...],
    max_attempts: int = 3,
    sleep: Any = None,
) -> dict[str, Any]:
    """POST with bounded retry on 429/5xx, size cap, and unconditional redaction on failure."""

    def attempt() -> Any:
        try:
            response = transport(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise LLMError(redact(f"request failed: {exc}", *secrets)) from None
        status = int(getattr(response, "status_code", 0) or 0)
        if _retryable(status):
            header = {}
            try:
                header = dict(getattr(response, "headers", {}) or {})
            except Exception:  # noqa: BLE001
                header = {}
            raw_retry = header.get("Retry-After") or header.get("retry-after")
            try:
                retry_after = float(raw_retry) if raw_retry is not None else None
            except (TypeError, ValueError):
                retry_after = None
            raise _RetryableStatus(status, _body_text(response), retry_after)
        if status >= 400:
            raise LLMError(
                redact(f"provider returned HTTP {status}: {_body_text(response)}", *secrets)
            )
        return response

    kwargs: dict[str, Any] = {
        "max_attempts": max_attempts,
        "base_delay": 1.0,
        "max_delay": 20.0,
        "retry_on": (_RetryableStatus,),
        "get_retry_after": _retry_after,
    }
    if sleep is not None:
        kwargs["sleep"] = sleep
    try:
        response = with_retry(attempt, **kwargs)
    except _RetryableStatus as exc:
        raise LLMError(
            redact(f"provider returned HTTP {exc.status}: {exc.body}", *secrets)
        ) from None

    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)) and len(content) > MAX_RESPONSE_BYTES:
        raise LLMError("provider response exceeded the size limit")
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise LLMError(redact(f"provider returned non-JSON: {exc}", *secrets)) from None
    if not isinstance(data, dict):
        raise LLMError("provider returned an unexpected payload shape")
    return data


def _openai_complete(
    request: LLMRequest, spec: ProviderSpec, transport: Transport, sleep: Any = None
) -> Completion:
    url = f"{resolve_base_url(spec, request.base_url)}/chat/completions"
    headers = {
        "Authorization": f"Bearer {request.api_key}",
        "Content-Type": "application/json",
        **request.extra_headers,
    }
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ],
        "max_completion_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    data = _post(
        url,
        headers=headers,
        payload=payload,
        timeout=request.timeout_s,
        transport=transport,
        secrets=(request.api_key,),
        sleep=sleep,
    )
    choices = data.get("choices") or []
    text = ""
    stop_reason = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some gateways return content parts
            text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        else:
            text = str(content or "")
        stop_reason = str(choices[0].get("finish_reason") or "")
    usage = data.get("usage") or {}
    return Completion(
        text=text,
        model=str(data.get("model") or request.model),
        provider=spec.id,
        input_tokens=_int_or_none(usage.get("prompt_tokens")),
        output_tokens=_int_or_none(usage.get("completion_tokens")),
        stop_reason=stop_reason,
    )


def _anthropic_complete(
    request: LLMRequest, spec: ProviderSpec, transport: Transport, sleep: Any = None
) -> Completion:
    url = f"{resolve_base_url(spec, request.base_url)}/messages"
    headers = {
        "x-api-key": request.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        **request.extra_headers,
    }
    payload: dict[str, Any] = {
        "model": request.model,
        "system": request.system,
        "messages": [{"role": "user", "content": request.user}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    data = _post(
        url,
        headers=headers,
        payload=payload,
        timeout=request.timeout_s,
        transport=transport,
        secrets=(request.api_key,),
        sleep=sleep,
    )
    blocks = data.get("content") or []
    text = "".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type", "text") == "text"
    )
    usage = data.get("usage") or {}
    return Completion(
        text=text,
        model=str(data.get("model") or request.model),
        provider=spec.id,
        input_tokens=_int_or_none(usage.get("input_tokens")),
        output_tokens=_int_or_none(usage.get("output_tokens")),
        stop_reason=str(data.get("stop_reason") or ""),
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def complete(
    request: LLMRequest,
    *,
    transport: Transport | None = None,
    sleep: Any = None,
) -> Completion:
    """Call the configured provider and return a normalized completion."""
    if not request.api_key.strip():
        raise LLMError("no API key configured for this provider")
    if not request.model.strip():
        raise LLMError("no model configured for this provider")
    spec = get_provider(request.provider)
    send = transport or _default_transport
    if spec.wire == "anthropic":
        return _anthropic_complete(request, spec, send, sleep)
    return _openai_complete(request, spec, send, sleep)


def catalog() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in PROVIDERS]
