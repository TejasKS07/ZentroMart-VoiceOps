"""Grounded generation with structural evidence and numeric-claim validation."""

from dataclasses import asdict, replace
from copy import deepcopy
from .language import language_of
import hashlib
import json
from pathlib import Path
from decimal import Decimal
import re
import time

from .agent import AgentOutputError, CompletionClient, parse_object
from .config import REFERENCE_DATE
from .redaction import normalize_digits
from .retrieval.index import KnowledgeIndex
from .schemas import AgentResponse, CustomerInput, Decision, GroundedClaim

RAG_PROMPT = (Path(__file__).parent / "prompts/rag.txt").read_text(encoding="utf-8").replace(
    "{reference_date}", REFERENCE_DATE.strftime("%d %B %Y"))
RAG_PROMPT_VERSION = "rag-v4"
RAG_PROMPT_SHA256 = hashlib.sha256(RAG_PROMPT.encode()).hexdigest()
RAG_RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["answer", "clarify", "refuse", "escalate"]},
        "reply": {"type": "string"}, "language": {"type": "string"}, "supported": {"type": "boolean"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["decision", "reply", "language", "supported", "evidence_ids"]}


def numeric_tokens(text: str) -> set[str]:
    # Date components such as 01 and decimal amounts such as 2.00 are the same
    # numeric values as 1 and 2. Retain the strict source check after normalization.
    return {format(Decimal(value).normalize(), 'f') for value in
            re.findall(r"\d+(?:\.\d+)?", normalize_digits(text).replace(",", ""))}


class RagAgent:
    def __init__(self, client: CompletionClient, index: KnowledgeIndex, *, top_k: int = 12, knowledge=None):
        self.client, self.index, self.top_k = client, index, top_k
        self.calculator = None
        if knowledge is not None:
            from .rules.calculations import PolicyCalculator
            from .rules.policies import PolicyRules
            self.calculator = PolicyCalculator(PolicyRules.from_knowledge(knowledge))

    def respond(self, customer: CustomerInput, *, previous_topic: str = "", delivered_on=None,
                operations=None, calculations=(), conversation_state=None) -> AgentResponse:
        started = time.perf_counter()
        verified = operations.gateway.order_facts(customer.session_id) if operations else None
        if verified and verified.get("delivered_on"):
            from datetime import date
            delivered_on = date.fromisoformat(verified["delivered_on"])
        found = self.index.search(customer.text, k=self.top_k, previous_topic=previous_topic, delivered_on=delivered_on)
        context = {e.evidence_id: e for e in found.context}
        from .rules.public_facts import dates_in_text, refund_calculations
        if not calculations:
            calculations = refund_calculations(customer.text, self.calculator)
        calculation_context = [{"values": c.values, "evidence_ids": list(c.evidence_ids)} for c in calculations]
        # Calculation citations must be original index sources (catalog/policy).
        for calculation in calculations:
            for source in calculation.source_evidence:
                context[source.evidence_id] = source
            for evidence_id in calculation.evidence_ids:
                if evidence_id in self.index.by_id:
                    context[evidence_id] = self.index.by_id[evidence_id]
        reply_language=language_of(customer.text,(conversation_state or {}).get("language","en"))
        payload = {"reply_language":reply_language,"customer_message": customer.text, "sources": [asdict(e) for e in context.values()],
                   "verified_order_facts": verified, "calculations": calculation_context}
        if conversation_state is not None:
            payload["conversation_state"] = conversation_state
        if operations:
            payload = operations.gateway.sanitize(customer.session_id, payload)
        retrieval_ms = (time.perf_counter()-started)*1000
        schema=deepcopy(RAG_RESPONSE_SCHEMA)
        if context:schema['properties']['evidence_ids']['items']['enum']=sorted(context)
        schema['properties']['language']['enum']=[reply_language]
        generation = self.client.complete([{"role": "system", "content": RAG_PROMPT},
                                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                          response_schema=schema)
        safe_generation = replace(generation, raw_output=operations.gateway.sanitize(customer.session_id, generation.raw_output)) if operations else generation
        try:
            if generation.finish_reason not in {"", "stop"}:
                raise ValueError("Incomplete generation")
            result = parse_object(generation.raw_output)
            if set(result) != {"decision", "reply", "language", "supported", "evidence_ids"}:
                raise ValueError("Invalid RAG response fields")
            if not all(isinstance(result[k], str) for k in ("decision", "reply", "language")) or type(result["supported"]) is not bool:
                raise ValueError("Invalid RAG response types")
            ids = result['evidence_ids']
            if not isinstance(ids,list) or not all(isinstance(i,str) and i in context for i in ids):
                raise ValueError('Unknown source citation')
            # Duplicate annotations have no factual meaning; preserve the raw
            # output and emit a single copy of each actually cited source.
            ids=list(dict.fromkeys(ids))
            if result['supported'] and not ids:raise ValueError('Supported reply needs evidence')
            if not result['supported'] and ids:raise ValueError('Unsupported reply must not claim evidence')
            from .rules.public_facts import complete_catalog_warranty
            original_reply=result['reply']
            result['reply']=complete_catalog_warranty(original_reply,customer.text,[context[i] for i in ids],reply_language)
            allowed_numbers=numeric_tokens(' '.join(context[i].text for i in ids))
            for calculation in calculation_context:
                if set(calculation['evidence_ids']) & set(ids):
                    allowed_numbers |= numeric_tokens(json.dumps(calculation['values']))
                    if 'initiated_on' in calculation['values']:
                        allowed_dates=dates_in_text(json.dumps(calculation['values'])) | {REFERENCE_DATE}
                        if dates_in_text(result['reply']) - allowed_dates:
                            raise ValueError('Date contradicts deterministic calculation')
            if verified:allowed_numbers |= numeric_tokens(json.dumps(verified))
            from .rules.public_facts import without_echoed_dates
            numeric_reply=without_echoed_dates(result['reply'],dates_in_text(customer.text))
            if numeric_tokens(numeric_reply)-allowed_numbers:
                raise ValueError('Reply contains unsupported numbers')
            if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z]{2,})?", result["language"]):
                raise ValueError("Invalid language code")
            reply = result["reply"]
            if operations and operations.gateway.sanitize(customer.session_id, reply) != reply:
                raise ValueError("Reply attempted to expose verification/contact data")
            return AgentResponse(Decision(result["decision"]), reply,
                evidence=tuple(context[i] for i in ids), retrieved_evidence_ids=tuple(context),
                language=result["language"], timings_ms={"retrieval": retrieval_ms, "llm": generation.latency_ms},
                generation=safe_generation, supported=result["supported"],
                claims=(GroundedClaim(reply,tuple(ids)),) if ids else (),
                diagnostics={'postprocessing':['catalog_warranty_duration']} if reply!=original_reply else {})
        except (ValueError, TypeError, KeyError):
            raise AgentOutputError("Model output failed grounded evidence/claim checks", safe_generation) from None
