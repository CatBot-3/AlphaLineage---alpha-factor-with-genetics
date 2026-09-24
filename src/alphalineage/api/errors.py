"""Errors shared by synchronous requests and background jobs."""

from __future__ import annotations

from typing import Any


class ActionError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        action: str = "Retry",
        symbols: list[str] | None = None,
        retryable: bool = False,
        details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = dict(
            code=code,
            message=message,
            action=action,
            symbols=symbols or [],
            retryable=retryable,
            details=details,
        )


def error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ActionError):
        return exc.payload
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        return {
            "code": "request_failed",
            "message": str(detail.get("message", detail)),
            "retryable": False,
            **detail,
        }
    return {
        "code": "request_failed" if detail else "operation_failed",
        "message": str(detail or str(exc) or "The operation could not be completed."),
        "retryable": False,
        "action": "Review details",
        "symbols": [],
        "details": type(exc).__name__,
    }
