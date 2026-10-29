"""Step 1: a stateless, prompt-only agent. No dataset or evaluator imports."""

import hashlib
import json
from pathlib import Path
import re
from typing import Protocol

from .config import REFERENCE_DATE
from .schemas import AgentResponse, CustomerInput, Decision, GenerationTrace

BASELINE_PROMPT_VERSION = "baseline-v1"
BASELINE_PROMPT = (Path(__file__).parent / "prompts/baseline.txt").read_text(encoding="utf-8").replace(
    "{reference_date}", REFERENCE_DATE.strftime("%d %B %Y"))
BASELINE_PROMPT_SHA256 = hashlib.sha256(BASELINE_PROMPT.encode()).hexdigest()


class CompletionClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> GenerationTrace: ...


class AgentOutputError(ValueError):
    def __init__(self, message: str, generation: GenerationTrace):
        super().__init__(message)
        self.generation = generation
        self.code = "invalid_agent_output"


def parse_object(text: str) -> dict:
    """Unwrap only a complete optional Markdown fence; never rewrite content."""
    content = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", content, flags=re.DOTALL)
    if fenced:
        content = fenced.group(1)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


class BaselineAgent:
    def __init__(self, client: CompletionClient):
        self.client = client

    def respond(self, customer: CustomerInput) -> AgentResponse:
        if not customer.text.strip():
            raise ValueError("Customer message must not be empty")
        generation = self.client.complete([
            {"role": "system", "content": BASELINE_PROMPT},
            {"role": "user", "content": customer.text},
        ])
        try:
            if generation.finish_reason not in {"", "stop"}:
                raise ValueError("Incomplete or blocked generation")
            data = parse_object(generation.raw_output)
            if set(data) != {"decision", "reply", "language"}:
                raise ValueError("Unexpected response fields")
            if not all(isinstance(value, str) for value in data.values()):
                raise ValueError("Response fields must be strings")
            if not re.fullmatch(r"[a-z]{2,3}(?:-[a-zA-Z]{2,})?", data["language"]):
                raise ValueError("Invalid language code")
            return AgentResponse(Decision(data["decision"]), data["reply"], language=data["language"],
                                 timings_ms={"llm": generation.latency_ms}, generation=generation)
        except (ValueError, TypeError, KeyError):
            raise AgentOutputError("Model output failed the baseline response contract", generation) from None
