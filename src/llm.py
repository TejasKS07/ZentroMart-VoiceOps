"""Text-only Chat Completions transport shared by smoke, agent and evaluator.

Retries transient transport failures, never repairs/re-generates model answers.
No audio input, web search, provider tools, policy files or evaluation labels.
"""

import time
from typing import Any

from .config import ConfigurationError, Settings
from .schemas import GenerationTrace


class LLMError(RuntimeError):
    def __init__(self, message: str, *, code: str = "llm_error"):
        super().__init__(message)
        self.code = code


class ChatCompletionClient:
    max_attempts = 3

    def __init__(self, settings: Settings):
        if not settings.llm_endpoint or not settings.llm_model:
            raise ConfigurationError("Set LLM_ENDPOINT and LLM_MODEL before making LLM requests")
        if not settings.llm_calls_enabled:
            raise ConfigurationError("LLM calls are deferred. Set LLM_CALLS_ENABLED=true only when ready for the final project evaluation.")
        self.settings = settings

    def complete(self, messages: list[dict[str, str]], *,
                 parameters: dict[str, Any] | None = None) -> GenerationTrace:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        params = self.settings.generation_parameters() if parameters is None else parameters
        if set(params) - {"temperature", "reasoning_effort"}:
            raise ValueError("Unsupported generation parameters")
        body = {"model": self.settings.llm_model, "messages": messages, **params}
        started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.post(self.settings.llm_endpoint, headers=headers, json=body,
                                         timeout=self.settings.timeout_seconds, allow_redirects=False)
            except requests.RequestException:
                if attempt == self.max_attempts:
                    raise LLMError("LLM connection failed; check endpoint/network/credentials", code="connection_failed") from None
                time.sleep(attempt)
                continue
            status = response.status_code
            if status in {408, 429, 500, 502, 503, 504} and attempt < self.max_attempts:
                time.sleep(attempt)
                continue
            if status != 200:
                raise LLMError(f"LLM request returned HTTP {status}", code=f"http_{status}")
            try:
                data = response.json()
                choice = data["choices"][0]
                text = choice["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError
                usage = data.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ValueError
                safe_usage = {k: v for k, v in usage.items()
                              if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                              and type(v) is int and v >= 0}
                details = usage.get("completion_tokens_details") or {}
                if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
                    safe_usage["reasoning_tokens"] = details["reasoning_tokens"]
                return GenerationTrace(
                    raw_output=text, model=str(data.get("model") or self.settings.llm_model),
                    response_id=str(data.get("id") or ""),
                    finish_reason=str(choice.get("finish_reason") or ""),
                    usage=safe_usage, latency_ms=(time.perf_counter()-started)*1000, attempts=attempt,
                )
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                raise LLMError("LLM response did not match the text Chat Completions contract", code="invalid_api_response") from None
        raise AssertionError("unreachable")
