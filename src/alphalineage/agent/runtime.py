"""P10-T4 - building the chat model the agent loop drives.

LangChain performs the HTTP here, which means it bypasses ``explain.providers``' egress
allowlist. That guard is therefore re-asserted in this module: the base URL is validated through
:func:`~alphalineage.explain.providers.resolve_base_url` *before* a client is constructed, so a
provider still cannot be pointed at a host the user did not choose. Key redaction is likewise
re-applied around every call, because a LangChain exception will happily include the request it
was making.

Imports are deliberately lazy. LangChain is a core dependency now (see ``docs/AGENT.md``
§2.1), but a broken or partially-installed LangChain must degrade to "the agent is unavailable",
not "the application will not start" — training, explanation, and the rest of the API have
nothing to do with it.
"""

from __future__ import annotations

from typing import Any

from alphalineage.explain.credentials import LLMConfig
from alphalineage.explain.providers import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    LLMError,
    get_provider,
    redact,
    resolve_base_url,
)


class AgentUnavailable(RuntimeError):
    """The agent runtime cannot start — usually a missing or broken LangChain install."""


def _require_langchain() -> tuple[Any, Any]:
    """Import the chat-model classes, turning an import failure into a clear message."""
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise AgentUnavailable(
            "the agent runtime needs langchain-openai and langchain-anthropic; "
            f"reinstall the package to repair it ({exc})"
        ) from None
    return ChatOpenAI, ChatAnthropic


def build_chat_model(
    config: LLMConfig,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_retries: int = 2,
) -> Any:
    """Construct a LangChain chat model for the user's configured provider.

    The egress allowlist is enforced here, not by LangChain: ``resolve_base_url`` raises before
    any client exists if the host is not one this provider is permitted to call.
    """
    if not config.key.is_set:
        raise LLMError(f"no API key configured for provider '{config.provider}'")
    if not config.model.strip():
        raise LLMError(f"no model configured for provider '{config.provider}'")

    spec = get_provider(config.provider)
    base_url = resolve_base_url(spec, config.base_url)  # the guard, before anything is built
    chat_openai, chat_anthropic = _require_langchain()

    common: dict[str, Any] = {
        "model": config.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_s,
        "max_retries": max_retries,
    }
    try:
        if spec.wire == "anthropic":
            return chat_anthropic(api_key=config.key.value, base_url=base_url, **common)
        return chat_openai(api_key=config.key.value, base_url=base_url, **common)
    except Exception as exc:  # noqa: BLE001 - never let a constructor echo the key
        message = redact(f"could not build the model client: {exc}", config.key.value)
        raise LLMError(message) from None


def supports_tool_calling(config: LLMConfig) -> tuple[bool, str]:
    """Whether the selected model can drive a tool loop at all.

    Reasoning-only endpoints are the trap here: DeepSeek's ``deepseek-reasoner`` does not accept
    function definitions, so an agent run against it fails in a confusing way partway through
    rather than at the start. Better to refuse up front and say why.
    """
    model = config.model.lower()
    if config.provider == "deepseek" and "reasoner" in model:
        return False, (
            "deepseek-reasoner does not support function calling; use deepseek-chat for agent "
            "runs, or switch provider"
        )
    return True, ""


def invoke_model(model: Any, messages: list[Any], *, secret: str = "") -> Any:
    """Call the model once, redacting the key from anything that comes back as an exception."""
    try:
        return model.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - a provider error must never carry the key onward
        raise LLMError(redact(f"model call failed: {exc}", secret)) from None


def usage_from(message: Any) -> tuple[int, int]:
    """Input/output token counts from a LangChain ``AIMessage``, when the provider reports them."""
    usage = getattr(message, "usage_metadata", None) or {}
    try:
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    except (AttributeError, TypeError, ValueError):
        return 0, 0
