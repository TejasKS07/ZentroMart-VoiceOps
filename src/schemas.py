"""Shared serializable contracts for text, voice, tools and evaluation."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import math
from typing import Any, Iterable

from .redaction import redact


class Decision(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class CustomerInput:
    text: str
    session_id: str


@dataclass(frozen=True)
class GenerationTrace:
    raw_output: str
    model: str
    response_id: str = ""
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0
    attempts: int = 1

    def to_dict(self, secrets: Iterable[str] = ()) -> dict[str, Any]:
        return redact(asdict(self), secrets)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not all(isinstance(v, str) and v.strip() for v in (self.evidence_id, self.source, self.text)):
            raise ValueError("Evidence needs a stable ID, source and nonempty chunk text")


@dataclass(frozen=True)
class GroundedClaim:
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ToolEvent:
    name: str
    status: str  # succeeded, failed, or verification_required
    details: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.status not in {"succeeded", "failed", "verification_required"}:
            raise ValueError("Invalid tool status")

    def to_dict(self) -> dict[str, Any]:
        # Never serialize raw tool arguments, private rows, summaries or errors.
        allowed = {"order_id", "ticket_id", "created", "verified", "found", "error_code"}
        safe = {key: value for key, value in self.details.items()
                if key in allowed and isinstance(value, (str, bool, int, type(None)))}
        return {"name": self.name, "status": self.status, "details": safe}


@dataclass(frozen=True)
class AgentResponse:
    decision: Decision
    reply: str
    evidence: tuple[Evidence, ...] = ()
    retrieved_evidence_ids: tuple[str, ...] = ()
    tool_events: tuple[ToolEvent, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)
    language: str = "en"
    generation: GenerationTrace | None = None
    claims: tuple[GroundedClaim, ...] = ()
    supported: bool | None = None
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self):
        object.__setattr__(self, "decision", Decision(self.decision))
        if not isinstance(self.reply, str) or not self.reply.strip():
            raise ValueError("Reply must be nonempty")
        if any(not math.isfinite(v) or v < 0 for v in self.timings_ms.values()):
            raise ValueError("Timings must be finite nonnegative milliseconds")
        ids = [e.evidence_id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Used evidence IDs must be unique")

    def to_dict(self, secrets: Iterable[str] = ()) -> dict[str, Any]:
        return redact({
            "schema_version": self.schema_version, "decision": self.decision.value,
            "reply": self.reply, "language": self.language,
            "evidence_ids": [e.evidence_id for e in self.evidence],
            "evidence": [asdict(e) for e in self.evidence],
            "retrieved_evidence_ids": list(self.retrieved_evidence_ids),
            "tool_events": [e.to_dict() for e in self.tool_events],
            "timings_ms": dict(self.timings_ms),
            "generation": self.generation.to_dict() if self.generation else None,
            "supported": self.supported,
            "claims": [asdict(claim) for claim in self.claims],
        }, secrets)
