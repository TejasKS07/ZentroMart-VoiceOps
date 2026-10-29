"""Grounded generation with structural evidence and numeric-claim validation."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import time

from .agent import AgentOutputError, CompletionClient, parse_object
from .config import REFERENCE_DATE
from .redaction import normalize_digits
from .retrieval.index import KnowledgeIndex
from .schemas import AgentResponse, CustomerInput, Decision, GroundedClaim

RAG_PROMPT = (Path(__file__).parent / "prompts/rag.txt").read_text(encoding="utf-8").replace(
    "{reference_date}", REFERENCE_DATE.strftime("%d %B %Y"))
RAG_PROMPT_VERSION = "rag-v1"
RAG_PROMPT_SHA256 = hashlib.sha256(RAG_PROMPT.encode()).hexdigest()


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", normalize_digits(text).replace(",", "")))


class RagAgent:
    def __init__(self, client: CompletionClient, index: KnowledgeIndex, *, top_k: int = 12):
        self.client, self.index, self.top_k = client, index, top_k

    def respond(self, customer: CustomerInput, *, previous_topic: str = "", delivered_on=None,
                operations=None, calculations=()) -> AgentResponse:
        started = time.perf_counter()
        verified = operations.gateway.order_facts(customer.session_id) if operations else None
        if verified and verified.get("delivered_on"):
            from datetime import date
            delivered_on = date.fromisoformat(verified["delivered_on"])
        found = self.index.search(customer.text, k=self.top_k, previous_topic=previous_topic, delivered_on=delivered_on)
        context = {e.evidence_id: e for e in found.context}
        calculation_context = [{"values": c.values, "evidence_ids": list(c.evidence_ids)} for c in calculations]
        # Calculation citations must be original index sources (catalog/policy).
        for calculation in calculations:
            for source in calculation.source_evidence:
                context[source.evidence_id] = source
            for evidence_id in calculation.evidence_ids:
                if evidence_id in self.index.by_id:
                    context[evidence_id] = self.index.by_id[evidence_id]
        payload = {"customer_message": customer.text, "sources": [asdict(e) for e in context.values()],
                   "verified_order_facts": verified, "calculations": calculation_context}
        if operations:
            payload = operations.gateway.sanitize(customer.session_id, payload)
        retrieval_ms = (time.perf_counter()-started)*1000
        generation = self.client.complete([{"role": "system", "content": RAG_PROMPT},
                                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        safe_generation = replace(generation, raw_output=operations.gateway.sanitize(customer.session_id, generation.raw_output)) if operations else generation
        try:
            if generation.finish_reason not in {"", "stop"}:
                raise ValueError("Incomplete generation")
            result = parse_object(generation.raw_output)
            if set(result) != {"decision", "reply", "language", "supported", "evidence_ids", "claims"}:
                raise ValueError("Invalid RAG response fields")
            if not all(isinstance(result[k], str) for k in ("decision", "reply", "language")) or type(result["supported"]) is not bool:
                raise ValueError("Invalid RAG response types")
            ids, claims = result["evidence_ids"], result["claims"]
            if not isinstance(ids, list) or not all(isinstance(i, str) and i in context for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("Unknown/duplicate source citation")
            if not isinstance(claims, list):
                raise ValueError("Invalid claims")
            if result["supported"] and not ids:
                raise ValueError("Supported reply needs evidence")
            if not result["supported"] and (ids or claims):
                raise ValueError("Unsupported reply must not claim evidence")
            if result["supported"] and result["decision"] == "answer" and not claims:
                raise ValueError("Factual answer needs claim provenance")
            for claim in claims:
                if not isinstance(claim, dict) or set(claim) != {"text", "evidence_ids"}:
                    raise ValueError("Invalid claim shape")
                if not isinstance(claim["text"], str) or not claim["text"] or claim["text"] not in result["reply"]:
                    raise ValueError("Claim must quote an actual reply substring")
                claim_ids = claim["evidence_ids"]
                if not isinstance(claim_ids, list) or not claim_ids or any(i not in ids for i in claim_ids):
                    raise ValueError("Claim cites unused evidence")
                allowed_numbers = numeric_tokens(" ".join(context[i].text for i in claim_ids))
                for calculation in calculation_context:
                    if set(calculation["evidence_ids"]) & set(claim_ids):
                        allowed_numbers |= numeric_tokens(json.dumps(calculation["values"]))
                if verified:
                    allowed_numbers |= numeric_tokens(json.dumps(verified))
                if numeric_tokens(claim["text"]) - allowed_numbers:
                    raise ValueError("Claim contains unsupported numbers")
            covered_numbers = set().union(*(numeric_tokens(claim["text"]) for claim in claims))
            if numeric_tokens(result["reply"]) - covered_numbers:
                raise ValueError("Numeric reply facts lack claim provenance")
            if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z]{2,})?", result["language"]):
                raise ValueError("Invalid language code")
            reply = result["reply"]
            if operations and operations.gateway.sanitize(customer.session_id, reply) != reply:
                raise ValueError("Reply attempted to expose verification/contact data")
            return AgentResponse(Decision(result["decision"]), reply,
                evidence=tuple(context[i] for i in ids), retrieved_evidence_ids=tuple(context),
                language=result["language"], timings_ms={"retrieval": retrieval_ms, "llm": generation.latency_ms},
                generation=safe_generation, supported=result["supported"],
                claims=tuple(GroundedClaim(c["text"], tuple(c["evidence_ids"])) for c in claims))
        except (ValueError, TypeError, KeyError):
            raise AgentOutputError("Model output failed grounded evidence/claim checks", safe_generation) from None
