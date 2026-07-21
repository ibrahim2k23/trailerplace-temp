from __future__ import annotations

from typing import Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src.config import settings
from src.llm import usage
from src.tracing import report_llm_usage, traceable_or_passthrough

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def structured(self, *, system: str, messages: list[dict], schema: type[T]) -> T:
        ...


class OpenAILLMClient:
    def __init__(self, model: str | None = None, reasoning_effort: str | None = None) -> None:
        self.model = model or settings.openai_model
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(api_key=settings.openai_api_key or None)

    @traceable_or_passthrough("llm.structured", run_type="llm")
    def structured(self, *, system: str, messages: list[dict], schema: type[T]) -> T:
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": schema,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        parsed = self._client.beta.chat.completions.parse(**request)
        # Counted for the M9 cost audit: Analyze + Respond on a normal turn.
        # Feature reranking records its separately tagged conditional third call.
        tokens = getattr(parsed, "usage", None)
        prompt_tokens = getattr(tokens, "prompt_tokens", 0) or 0
        completion_tokens = getattr(tokens, "completion_tokens", 0) or 0
        usage.record_completion(self.model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        # Feed the same counts to LangSmith so the run shows Tokens/Cost (the local
        # audit and the dashboard read from one place).
        report_llm_usage(self.model, prompt_tokens, completion_tokens)
        result = parsed.choices[0].message.parsed
        if result is None:
            raise RuntimeError("OpenAI returned no parsed structured output")
        return result
