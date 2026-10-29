"""Verification and idempotent ticket creation around unmodified supplied tools."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from types import ModuleType

from ..redaction import normalize_digits, redact
from ..schemas import ToolEvent


class VerificationError(ValueError):
    pass


def identifier(value: str, lengths: tuple[int, ...]) -> str:
    normalized = normalize_digits(str(value))
    if re.search(r"[^0-9\s-]", normalized):
        raise ValueError("Identifier must contain only digits, spaces or hyphens")
    digits = re.sub(r"[\s-]", "", normalized)
    if len(digits) not in lengths:
        raise ValueError("Identifier has an invalid length")
    return digits


class SuppliedTools:
    def __init__(self, dataset_dir: Path, runtime_dir: Path):
        dataset_dir, runtime_dir = dataset_dir.resolve(), runtime_dir.resolve()
        if runtime_dir.is_relative_to(dataset_dir):
            raise ValueError("Tool outputs must be outside the dataset")
        script = dataset_dir / "tools/tools.py"
        # Isolated module namespace, with no bytecode or ticket writes in Dataset.
        self._module = ModuleType("voiceops_supplied_tools")
        self._module.__file__ = str(script)
        exec(compile(script.read_bytes(), str(script), "exec"), self._module.__dict__)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._module._TICKETS = str(runtime_dir / "tickets_created.csv")

    def lookup_order(self, order_id: str, verification_value: str | None = None) -> dict:
        return self._module.lookup_order(order_id, verification_value=verification_value)

    def create_ticket(self, summary: str, **kwargs) -> dict:
        return self._module.create_ticket(summary, **kwargs)


@dataclass(repr=False)
class OrderState:
    attempts: int = 0
    verified: bool = False
    facts: dict = field(default_factory=dict, repr=False)
    proofs: list[str] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    order_id: str
    attempts_remaining: int
    event: ToolEvent

    @property
    def verified(self) -> bool:
        return self.status == "verified"


class ToolGateway:
    def __init__(self, dataset_dir: Path, runtime_dir: Path, *, backend=None):
        dataset_dir, runtime_dir = dataset_dir.resolve(), runtime_dir.resolve()
        if runtime_dir.is_relative_to(dataset_dir):
            raise ValueError("Tool outputs must be outside the dataset")
        self.backend = backend or SuppliedTools(dataset_dir, runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._ledger = runtime_dir / "ticket_idempotency.sqlite3"
        with sqlite3.connect(self._ledger) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS tickets (key TEXT PRIMARY KEY, ticket_id TEXT NOT NULL)")
        self._states: dict[str, dict[str, OrderState]] = {}
        self._active: dict[str, str] = {}
        self._lock = RLock()

    def _state(self, session_id: str) -> tuple[str, OrderState]:
        if session_id not in self._active:
            raise VerificationError("Obtain an order ID first")
        order_id = self._active[session_id]
        return order_id, self._states[session_id][order_id]

    def begin(self, session_id: str, order_id: str) -> VerificationResult:
        if not session_id.strip():
            raise ValueError("Session ID must not be empty")
        order_id = identifier(order_id, (8,))
        with self._lock:
            state = self._states.setdefault(session_id, {}).setdefault(order_id, OrderState())
            self._active[session_id] = order_id
            if state.verified:
                return VerificationResult("verified", order_id, 0,
                    ToolEvent("lookup_order", "succeeded", {"order_id": order_id, "verified": True}))
            if state.attempts >= 2:
                return VerificationResult("locked", order_id, 0,
                    ToolEvent("lookup_order", "failed", {"order_id": order_id, "error_code": "verification_locked"}))
            # Always exercise the supplied identity-gated lookup; its response
            # never establishes whether this order exists.
            try:
                self.backend.lookup_order(order_id)
            except Exception:
                return VerificationResult("tool_error", order_id, 2-state.attempts,
                    ToolEvent("lookup_order", "failed", {"order_id": order_id, "error_code": "lookup_failed"}))
            return VerificationResult("verification_required", order_id, 2-state.attempts,
                ToolEvent("lookup_order", "verification_required", {"order_id": order_id}))

    def verify(self, session_id: str, caller_proof: str) -> VerificationResult:
        with self._lock:
            order_id, state = self._state(session_id)
            if state.verified:
                return VerificationResult("verified", order_id, 0,
                    ToolEvent("lookup_order", "succeeded", {"order_id": order_id, "verified": True}))
            if state.attempts >= 2:
                return VerificationResult("locked", order_id, 0,
                    ToolEvent("lookup_order", "failed", {"order_id": order_id, "error_code": "verification_locked"}))
            state.attempts += 1
            state.proofs.append(str(caller_proof))
            try:
                proof = identifier(caller_proof, (4, 6))
            except ValueError:
                result = {"verified": False}
            else:
                state.proofs.append(proof)
                try:
                    result = self.backend.lookup_order(order_id, verification_value=proof)
                except Exception:
                    state.attempts -= 1
                    return VerificationResult("tool_error", order_id, 2-state.attempts,
                        ToolEvent("lookup_order", "failed", {"order_id": order_id, "error_code": "lookup_failed"}))
            if result.get("found") is True and result.get("verified") is True and str(result.get("order_id")) == order_id:
                state.verified = True
                state.facts = deepcopy(result)
                status = "verified"
            else:
                status = "locked" if state.attempts >= 2 else "verification_failed"
            return VerificationResult(status, order_id, max(0, 2-state.attempts),
                ToolEvent("lookup_order", "succeeded" if state.verified else "failed",
                          {"order_id": order_id, "verified": state.verified}))

    def order_facts(self, session_id: str) -> dict:
        with self._lock:
            _, state = self._state(session_id)
            if not state.verified:
                raise VerificationError("Order details require successful caller verification")
            forbidden = {"phone", "phone_last4", "email", "address", "street_address", "delivery_address", "delivery_pincode"}
            # No expected verification values reach the language model, even after
            # success. Pincode availability is a separate public-table lookup.
            return deepcopy({k: v for k, v in state.facts.items() if k not in forbidden})

    def sanitize(self, session_id: str, value):
        states = self._states.get(session_id, {})
        proofs = tuple(proof for state in states.values() for proof in state.proofs)
        result = redact(value, proofs)
        def clean(item):
            if isinstance(item, dict):
                return {key: clean(v) for key, v in item.items()}
            if isinstance(item, list):
                return [clean(v) for v in item]
            if isinstance(item, str):
                item = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[REDACTED]", item)
                normalized = normalize_digits(item)
                for match in reversed(list(re.finditer(r"(?<!\d)(?:\+91[\s-]*)?[5-9](?:[\s-]*\d){9}(?!\d)", normalized))):
                    item = item[:match.start()] + "[REDACTED]" + item[match.end():]
            return item
        return clean(result)

    def handover(self, session_id: str, *, issue_key: str, issue: str, intent: str,
                 checked: str, reason: str) -> ToolEvent:
        if not all(isinstance(v, str) and v.strip() for v in (session_id, issue_key, issue, intent, checked, reason)):
            raise ValueError("Handover requires issue, intent, checks and escalation reason")
        with self._lock:
            order_id = self._active.get(session_id)
            verified = bool(order_id and self._states[session_id][order_id].verified)
            summary = self.sanitize(session_id, f"Issue: {issue}; Intent: {intent}; Order: {order_id or 'not provided'}; "
                                    f"Identity verified: {verified}; Checks: {checked}; Escalation reason: {reason}.")
            key = hashlib.sha256(json.dumps([session_id, order_id, issue_key, reason]).encode()).hexdigest()
            with sqlite3.connect(self._ledger, timeout=30) as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute("SELECT ticket_id FROM tickets WHERE key=?", (key,)).fetchone()
                if existing:
                    return ToolEvent("create_ticket", "succeeded", {"created": True, "ticket_id": existing[0]})
                try:
                    result = self.backend.create_ticket(summary, order_id=order_id, reason=self.sanitize(session_id, reason),
                                                        intent=self.sanitize(session_id, intent))
                except Exception:
                    return ToolEvent("create_ticket", "failed", {"created": False, "error_code": "ticket_write_failed"})
                if result.get("created") is not True or not isinstance(result.get("ticket_id"), str) or not result["ticket_id"]:
                    return ToolEvent("create_ticket", "failed", {"created": False, "error_code": "ticket_not_created"})
                connection.execute("INSERT INTO tickets VALUES (?,?)", (key, result["ticket_id"]))
                return ToolEvent("create_ticket", "succeeded", {"created": True, "ticket_id": result["ticket_id"]})

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)
            self._active.pop(session_id, None)
