"""Central LLM client factory and safe-invoke wrapper.

Consolidates the model/temperature resolution that was previously copy-pasted
into ~25 ``@lru_cache`` factory functions across graph.py, service.py and the
mini classifiers, and adds one place to handle structured-output validation
failures.

Migration is intentionally behavior-preserving: ``make_llm`` reproduces the exact
resolution each site already used — a site-specific env var (if any), then the
global ``OPENAI_MODEL``, then the default — so callers can be switched over one at
a time without changing which model/temperature they run on. Callers keep their
own ``@lru_cache`` wrapper, so client construction stays cached per role.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"

T = TypeVar("T")


def resolve_model(model_env: str | None = None, *, default: str = DEFAULT_MODEL) -> str:
    """Resolve a model name: site-specific env var, then OPENAI_MODEL, then default."""
    return (
        (os.getenv(model_env) if model_env else None)
        or os.getenv("OPENAI_MODEL")
        or default
    ).strip()


def make_llm(
    *,
    model_env: str | None = None,
    default_model: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
    structured_output: Type[BaseModel] | None = None,
    method: str = "function_calling",
    **client_kwargs: Any,
):
    """Build a ChatOpenAI client (optionally with structured output).

    Passing ``temperature=None`` leaves it unset (used by reasoning models that
    reject an explicit temperature). Extra ``client_kwargs`` (e.g.
    ``use_responses_api``, ``reasoning``) are forwarded unchanged.
    """
    model = resolve_model(model_env, default=default_model)
    kwargs: dict[str, Any] = dict(client_kwargs)
    if temperature is not None:
        kwargs["temperature"] = temperature
    else:
        kwargs.setdefault("temperature", None)
    llm = ChatOpenAI(model=model, **kwargs)
    if structured_output is not None:
        return llm.with_structured_output(structured_output, method=method)
    return llm


def safe_invoke(llm: Any, messages: Any, *, fallback: T, role: str = "") -> T:
    """Invoke an LLM and return ``fallback`` on any failure.

    Structured-output calls can raise ``pydantic.ValidationError`` when the model
    emits an off-schema enum (function-calling enforces shape, not enum
    membership). This wrapper turns that — and any other exception — into a
    logged, safe fallback so callers do not each need their own try/except.
    """
    try:
        return llm.invoke(messages)
    except ValidationError as exc:
        logger.warning("structured_output_validation_error | role=%s | %s", role, exc)
        return fallback
    except Exception:
        logger.exception("llm_invoke_failed | role=%s", role)
        return fallback
