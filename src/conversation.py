"""Explicit, session-scoped support controller shared by text and future voice.

Routing and operational replies run locally. Optional RagAgent handles open policy
questions, using a minimal relevant context rather than a transcript with secrets.
"""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
import re
from threading import RLock
import time

from .redaction import normalize_digits
from .schemas import AgentResponse, CustomerInput, Decision, GroundedClaim
from .tools.gateway import VerificationError
from .tools.operations import SupportOperations


@dataclass(repr=False)
class SessionState:
    language: str = "en"
    introduced: bool = False
    active_order: str | None = None
    verification_attempts: int = 0
    verified: bool = False
    verified_facts: dict = field(default_factory=dict)
    intent: str = ""
    missing_fields: tuple[str, ...] = ()
    last_answer: str = ""
    evidence_ids: tuple[str, ...] = ()
    selected_line: str | None = None
    caller_fields: dict = field(default_factory=dict)
    issue_messages: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    tickets: dict[str, dict] = field(default_factory=dict)
    turns: int = 0
    repair_count: int = 0


def language_of(text: str, previous: str) -> str:
    if re.search(r"(?:in|speak|reply in) english|angrezi|अंग्रेज़ी|अंग्रेजी", text, re.I):
        return "en"
    if re.search(r"hindi mein|हिंदी में|हिन्दी में", text, re.I):
        return "hi" if re.search(r"[\u0900-\u097f]", text) else "hi-en"
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    if re.search(r"\b(mujhe|mera|meri|maine|hai|hain|ka|ki|kya|kab|kitne|aayega|sawaal|theek|achha|ise|kar|nahi|teesri)\b", text, re.I):
        return "hi-en"
    if re.search(r"\b(i|my|the|how|what|can|please|hello|hi|thanks|order|refund|cancel)\b", text, re.I):
        return "en"
    return previous


def detect_intent(text: str) -> str:
    # More specific intents precede broad order/return/refund mentions.
    patterns = (
        ("safety", r"swelling|swollen|burning|sparks?|injur|overheat|battery.*(?:hot|heat)|आग|चिंगारी|जलने|बैटरी.*(?:फूल|गर्म)|battery.*phool"),
        ("unauthorized", r"unauthori[sz]ed|fraud|hacked|did(?:n't| not) (?:make|place|authori[sz]e)|not my (?:order|payment|transaction)|(?:मैंने|maine).*(?:order|payment|ऑर्डर|भुगतान).*(?:नहीं|nahi)"),
        ("legal", r"legal (?:action|notice)|consumer court|police|lawsuit|sue\b|कानूनी|अदालत|पुलिस|consumer forum"),
        ("human", r"human|speak to a person|real person|live agent|representative|insaan|इंसान|मानव|सहयोगी"),
        ("repeat_contact", r"third (?:time|call)|(?:3rd|third|fourth).*contact|teesri|तीसरी|तीसरा|again.*(?:week|days)|बार.*(?:हफ्ते|सप्ताह)"),
        ("damage", r"damag|defect|wrong (?:item|product)|broken|टूट|खराब|क्षतिग्रस्त|गलत.*(?:सामान|वस्तु)|toot"),
        ("compensation", r"compensation|voucher.*(?:want|chahiye)|(?:want|chahiye).*voucher|rupaye.*voucher|मुआवज़|वाउचर"),
        ("express_credit", r"express.*(?:late|fee|credit)|(?:late|fee).*express|एक्सप्रेस.*(?:देरी|शुल्क)"),
        ("coupon", r"coupon|कूपन"),
        ("cancel", r"cancel|कैंसिल|रद्द"),
        ("delivery_absence", r"not at home|nobody.*home|miss.*delivery|घर.*नहीं|ghar.*nahi"),
        ("refund", r"refund|रिफ़ंड|रिफंड|पैसे.*वापस"),
        ("return", r"return|वापस|लौटा"),
        ("warranty", r"warranty|वारंटी|guarantee|गारंटी"),
        ("membership", r"\bplus\b|membership|मेंबरशिप|सदस्यता"),
        ("serviceability", r"serviceable|pincode|pin code|पिनकोड|पिन कोड"),
        ("new_order", r"\bnew order\b|\bplace (?:an? |the )?order\b|\border (?:place|karna|karo)\b|नया ऑर्डर|नए ऑर्डर|ऑर्डर.*(?:करना|करें)"),
        ("tracking", r"ship|arriv|where.*order|order.*(?:where|kab|kahan)|deliv|डिलीवर|शिप|ऑर्डर.*कहाँ|कब.*(?:आए|भेज)"),
        ("order", r"order|ऑर्डर"),
    )
    return next((intent for intent, pattern in patterns if re.search(pattern, text, re.I)), "")


def digit_values(text: str, length: int) -> list[str]:
    text = normalize_digits(text)
    values = []
    for match in re.finditer(r"\d+(?:[ -]+\d+)*", text):
        # Reject the entire maximal run when attached to letters. Do not trim an
        # invalid identifier into a shorter order ID or verification proof.
        before = text[match.start()-1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if before and (before.isalnum() or before == "_") or after and (after.isalnum() or after == "_"):
            continue
        value = re.sub(r"[ -]", "", match[0])
        if len(value) == length:
            values.append(value)
    return values


class ConversationAgent:
    def __init__(self, operations: SupportOperations, *, rag=None):
        self.operations, self.rag = operations, rag
        self._sessions: dict[str, SessionState] = {}
        self._lock = RLock()
        self._sources = {e.evidence_id: e for d in operations.knowledge.documents for e in d.sections}

    def snapshot(self, session_id: str) -> SessionState:
        with self._lock:
            return deepcopy(self._sessions.get(session_id, SessionState()))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self.operations.gateway.clear_session(session_id)

    def mark_introduced(self, session_id: str) -> None:
        """Record a voice recovery disclosure even when no text turn was routed."""
        with self._lock:
            self._sessions.setdefault(session_id,SessionState()).introduced=True

    @staticmethod
    def _say(state, en, hi, hinglish=None):
        return hi if state.language == "hi" else (hinglish or en) if state.language == "hi-en" else en

    def _reply(self, state, decision, en, hi, hinglish=None, *, ids=(), events=(), code=""):
        text = self._say(state, en, hi, hinglish)
        evidence = tuple(self._sources[i] for i in dict.fromkeys(ids) if i in self._sources)
        return AgentResponse(Decision(decision), text, evidence=evidence, language=state.language,
                             tool_events=tuple(events), supported=bool(evidence), diagnostics={"code": code} if code else {},
                             claims=(GroundedClaim(text, tuple(e.evidence_id for e in evidence)),) if evidence else ())

    def respond(self, customer: CustomerInput) -> AgentResponse:
        if not customer.session_id.strip() or not customer.text.strip():
            raise ValueError("Session ID and customer message must be nonempty")
        started = time.perf_counter()
        with self._lock:
            state = self._sessions.setdefault(customer.session_id, SessionState())
            state.language = language_of(customer.text, state.language)
            response = self._route(customer, state)
            gateway = self.operations.gateway
            safe = gateway.sanitize(customer.session_id, response.to_dict())
            if safe["reply"] != response.reply:
                response = self._reply(state, "refuse", "I cannot share verification or contact details; I can help with general policies.",
                    "मैं सत्यापन या संपर्क विवरण साझा नहीं कर सकता; सामान्य नीतियों में मदद कर सकता हूँ।", ids=("IDV-4",))
            if response.generation:
                response = replace(response, generation=replace(response.generation,
                    raw_output=gateway.sanitize(customer.session_id, response.generation.raw_output)))
            # Apply disclosure after the privacy fallback, so it cannot be lost.
            if not state.introduced:
                intro = self._say(state, "I'm ZentroMart's AI support assistant.",
                                  "मैं ZentroMart का AI सपोर्ट असिस्टेंट हूँ।", "Main ZentroMart ka AI support assistant hoon.")
                if not re.search(r"\bAI\b|artificial intelligence", response.reply, re.I):
                    response = replace(response, reply=intro + " " + response.reply)
                state.introduced = True
            state.last_answer = response.reply
            state.evidence_ids = tuple(e.evidence_id for e in response.evidence)
            state.turns += 1
            code = response.diagnostics.get("code") or self._result_code(response, state)
            status = "failed" if any(e.status == "failed" and e.details.get("error_code") in {"lookup_failed", "ticket_write_failed", "ticket_not_created"} for e in response.tool_events) else "needs_input" if response.decision == Decision.CLARIFY else "limited" if code in {"offline_unsupported", "offline_source_extract", "order_placement_unavailable"} else "completed"
            diagnostics = {"code": code, "status": status, "intent": state.intent or "unrecognized",
                "waiting_for": list(state.missing_fields), "verified": state.verified,
                "verification_attempts_used": state.verification_attempts,
                "mode": "live_policy" if self.rag else "offline", "llm_used": response.generation is not None,
                "tools": [{"name": e.name, "status": e.status} for e in response.tool_events],
                "evidence_ids": list(state.evidence_ids)}
            return replace(response, diagnostics=gateway.sanitize(customer.session_id, diagnostics),
                           timings_ms={**response.timings_ms, "conversation": (time.perf_counter()-started)*1000})

    @staticmethod
    def _result_code(response, state):
        if any(e.name == "create_ticket" for e in response.tool_events):
            return "ticket_created" if response.tool_events[-1].details.get("created") is True else "ticket_creation_failed"
        if any(e.name == "lookup_order" for e in response.tool_events):
            if any(e.details.get("error_code") == "lookup_failed" for e in response.tool_events):
                return "lookup_failed"
            return "order_verified" if state.verified else "verification_locked" if state.verification_attempts >= 2 else "verification_failed" if state.verification_attempts else "verification_required"
        return "clarification_required" if response.decision == Decision.CLARIFY else "request_refused" if response.decision == Decision.REFUSE else "grounded_answer" if response.evidence else "unsupported_answer"

    def _sync(self, session_id, state):
        gate = self.operations.gateway.verification_state(session_id)
        state.active_order, state.verified = gate["order_id"], gate["verified"]
        state.verification_attempts = gate["attempts"]
        state.verified_facts = self.operations.gateway.order_facts(session_id) if state.verified else {}

    def _new_order(self, state, text):
        if state.intent != "new_order":
            self._reset_issue(state)
        state.repair_count = 0
        if re.search(r"\b(?:existing|already)\b|पहले", text, re.I):
            state.intent = "order"
            state.missing_fields = ("order_id",)
            return self._reply(state, "clarify", "Please enter the 8-digit order ID shown in My Orders.",
                "My Orders में दिखने वाला 8 अंकों का ऑर्डर नंबर दें।", "My Orders mein dikhne wala 8-digit order ID dein.", code="order_id_required")
        state.intent = "new_order"
        if re.search(r"\b(?:place|buy|purchase)\b|order (?:karna|karo)|खरीद|करना", text, re.I):
            state.missing_fields = ("shopping_question",)
            return self._reply(state, "clarify", "I can help with product and delivery questions, but placing an order isn't available in this chat. What would you like to know before ordering?",
                "उत्पाद और डिलीवरी के सवालों में मदद कर सकता हूँ, पर इस चैट में ऑर्डर देना उपलब्ध नहीं है। ऑर्डर देने से पहले क्या जानना चाहेंगे?",
                "Product aur delivery questions mein help kar sakta hoon, par is chat mein order place nahi hota. Order se pehle kya jaanna hai?", code="order_placement_unavailable")
        state.missing_fields = ("order_goal",)
        return self._reply(state, "clarify", "Do you want to place a new order, or get help with an existing order?",
            "नया ऑर्डर देना चाहते हैं या पहले से दिए ऑर्डर में मदद चाहिए?", "Naya order place karna hai ya existing order mein help chahiye?", code="order_goal_ambiguous")

    def _input_guidance(self, state):
        state.repair_count += 1
        if state.missing_fields == ("order_goal",):
            return self._reply(state, "clarify", "Are you buying something, or checking an order you've already placed? You can reply 'place an order' or 'existing order'.",
                "आप खरीदना चाहते हैं या पहले दिए ऑर्डर की जाँच करना चाहते हैं? 'नया ऑर्डर' या 'पहले का ऑर्डर' लिख सकते हैं।",
                "Kuch khareedna hai ya already placed order check karna hai? 'Place an order' ya 'existing order' likh sakte hain.", code="order_goal_ambiguous")
        if state.missing_fields == ("order_id",):
            state.intent = "order"
            state.missing_fields = ("order_id",)
            return self._reply(state, "clarify", "I still need the 8-digit order ID shown in My Orders. If you don't have it, you can ask a general policy question or request a human agent.",
                "My Orders में दिखने वाला 8 अंकों का ऑर्डर नंबर चाहिए। नंबर न हो तो सामान्य नीति पूछ सकते हैं या मानव सहयोगी की मदद माँग सकते हैं।",
                "My Orders wala 8-digit order ID chahiye. ID na ho toh general policy pooch sakte hain ya human agent maang sakte hain.", code="order_id_required")
        if state.missing_fields == ("verification",):
            return self._reply(state, "clarify", "I'm waiting for one verification value: your registered phone's last four digits or this order's delivery pincode. Which one can you provide?",
                "सत्यापन के लिए रजिस्टर्ड फ़ोन के आख़िरी चार अंक या इस ऑर्डर का डिलीवरी पिनकोड चाहिए। आप कौन सा दे सकते हैं?",
                "Verification ke liye registered phone ke last four digits ya is order ka delivery pincode chahiye. Kaunsa de sakte hain?", code="verification_required")
        state.missing_fields = ("intent",)
        if state.repair_count == 1:
            return self._reply(state, "clarify", "I couldn't identify the support issue yet. Is this about delivery, a return, a refund, or cancellation?",
                "अभी आपकी समस्या समझ नहीं आई। क्या यह डिलीवरी, रिटर्न, रिफ़ंड या कैंसिल करने से जुड़ी है?",
                "Abhi support issue samajh nahi aaya. Delivery, return, refund ya cancellation mein help chahiye?", code="input_not_understood")
        if state.repair_count >= 3:
            return self._reply(state, "clarify", "I still can't identify the issue. Type 'human agent' for a handover, or describe the problem in one sentence.",
                "समस्या अभी भी समझ नहीं आई। मानव सहयोगी के लिए 'मानव एजेंट' लिखें या एक वाक्य में समस्या बताएं।",
                "Issue abhi bhi samajh nahi aaya. Handover ke liye 'human agent' likhein ya ek sentence mein problem batayein.", code="input_not_understood")
        return self._reply(state, "clarify", "I'm still missing the issue; try 'track my order' or 'explain the refund policy'. What would you like to do?",
            "समस्या अभी साफ़ नहीं है; जैसे 'मेरा ऑर्डर कहाँ है' या 'रिफ़ंड नीति बताएं' लिखें। आप क्या करना चाहते हैं?",
            "Issue abhi clear nahi hai; 'track my order' ya 'refund policy batao' likh sakte hain. Kya karna chahte hain?", code="input_not_understood")

    def _route(self, customer, state):
        sid, text = customer.session_id, customer.text
        gateway = self.operations.gateway
        intent = detect_intent(text)
        orders = digit_values(text, 8)
        proof_values = digit_values(text, 4) + digit_values(text, 6)
        explicit_proof = bool(re.search(r"last.*(?:four|4)|phone|pincode|pin code|digits|अंक|पिनकोड|पिन कोड", text, re.I))
        if intent == "serviceability" and state.active_order and not state.verified and re.search(r"\d", normalize_digits(text)):
            # A pincode supplied for identity must not change the pending issue.
            intent = ""
        begin_event = None
        if proof_values and ((explicit_proof and intent != "serviceability") or state.missing_fields == ("verification",)):
            for proof in proof_values:
                gateway.remember_proof(sid, proof)
        if len(set(orders)) > 1:
            state.missing_fields = ("order_id",)
            return self._reply(state, "clarify", "Which one order should we discuss first?", "पहले किस एक ऑर्डर पर बात करें?", "Pehle kis ek order par baat karein?")
        # Validate the pending field before trying to interpret it as another
        # identifier. Four/six digits here are not verification attempts.
        needs_order = state.missing_fields in {("order_id",), ("order_goal",)}
        order_candidate = not orders and (needs_order or intent == "order") and bool(re.search(r"\d", normalize_digits(text)))
        if order_candidate and intent in {"", "order", "new_order"}:
            state.intent = "order"
            state.missing_fields = ("order_id",)
            state.repair_count += 1
            count = len(re.sub(r"\D", "", normalize_digits(text)))
            if count == 8:
                return self._reply(state, "clarify", "An order ID must contain only 8 digits, optionally grouped with spaces or hyphens. Please copy the full ID from My Orders.",
                    "ऑर्डर नंबर में केवल 8 अंक हों; बीच में स्पेस या हाइफ़न दे सकते हैं। My Orders से पूरा नंबर दें।",
                    "Order ID mein sirf 8 digits hone chahiye; spaces ya hyphens se group kar sakte hain. My Orders se poora ID dein.", code="invalid_order_id")
            return self._reply(state, "clarify", f"That has {count} digits; an order ID needs exactly 8 digits. Please enter the full order ID shown in My Orders.",
                f"इसमें {count} अंक हैं; ऑर्डर नंबर ठीक 8 अंकों का होना चाहिए। My Orders में दिखने वाला पूरा ऑर्डर नंबर दें।",
                f"Ismein {count} digits hain; order ID exactly 8 digits ka hona chahiye. My Orders mein dikhne wala poora order ID dein.", code="invalid_order_id")
        if state.missing_fields == ("verification",) and not orders and not proof_values and not intent and re.search(r"\d", normalize_digits(text)):
            return self._reply(state, "clarify", "Verification needs your registered phone's last 4 digits or the delivery pincode's 6 digits. Please send one complete value.",
                "सत्यापन के लिए रजिस्टर्ड फ़ोन के आख़िरी 4 अंक या डिलीवरी पिनकोड के 6 अंक चाहिए। एक पूरा मान दें।",
                "Verification ke liye phone ke last 4 digits ya delivery pincode ke 6 digits chahiye. Ek complete value dein.", code="invalid_verification_format")
        if intent == "new_order" or state.missing_fields == ("order_goal",) and re.search(r"\b(?:place|buy|purchase|existing|already)\b|खरीद|पहले", text, re.I):
            return self._new_order(state, text)
        vague = bool(re.fullmatch(r"[\s?!.,]*(?:what|wjhat|wat|huh|help|hello|hi|hey|kya|क्या|मदद)?[\s?!.,]*", text, re.I))
        if not orders and not intent and (vague or needs_order):
            return self._input_guidance(state)
        if intent or orders:
            state.repair_count = 0
        new_unspecified_order = re.search(r"another order|different order|other order|ek aur order|दूसर[ेा] ऑर्डर", text, re.I)
        if new_unspecified_order and not orders:
            gateway.deactivate(sid)
            self._reset_issue(state)
            self._sync(sid, state)
            state.intent = "order"
            state.missing_fields = ("order_id",)
            return self._reply(state, "clarify", "What is the new order ID?", "नए ऑर्डर का नंबर क्या है?", "Naye order ka number kya hai?")
        if orders and orders[0] != state.active_order:
            self._reset_issue(state)
            result = gateway.begin(sid, orders[0])
            begin_event = result.event
            self._sync(sid, state)
            state.intent = intent if intent not in {"", "order"} else "order"
            state.checks.append("Unverified order lookup")
            # Urgent handover never depends on disclosing private order facts.
            if intent in {"safety", "unauthorized", "legal", "human"}:
                self._record_issue(sid, state, text)
                return self._escalate(sid, state, [self._reason(intent)], events=(result.event,))
            if not proof_values:
                if result.verified:
                    response = self._answer(sid, state, text)
                    return replace(response, tool_events=(result.event,) + response.tool_events)
                return self._verification_reply(state, result)
        # New topical request replaces missing fields and unrelated issue history.
        if intent and intent != state.intent and intent not in {"order", "repeat_contact", "human", "legal", "safety", "unauthorized"}:
            self._reset_issue(state)
            state.intent = intent
        elif intent and not state.intent:
            state.intent = intent
        if intent == "order" and orders and state.intent != "order":
            self._reset_issue(state)
            state.intent = "order"
        if intent == "repeat_contact":
            state.intent = intent
        if intent in {"safety", "unauthorized", "legal", "human"}:
            self._record_issue(sid, state, text)
            return self._escalate(sid, state, [self._reason(intent)])
        if intent == "damage" and not state.verified:
            stated_value = re.search(r"(?:₹|rs\.?\s*|rupees\s*)([\d,]+)|([\d,]+)\s*(?:rupees|rupaye|रुपये)", normalize_digits(text), re.I)
            if stated_value:
                value = Decimal((stated_value[1] or stated_value[2]).replace(",", ""))
                reasons = self.operations.calculator.escalation_reasons(damaged=True, item_value=value).values["reasons"]
                if reasons:
                    self._record_issue(sid, state, text)
                    state.checks.append("Caller reports high-value damage; value is not order-verified")
                    return self._escalate(sid, state, reasons)
        self._record_issue(sid, state, text)
        # Pending identity only consumes a proof-shaped message, never a policy query.
        if proof_values and not state.verified and state.active_order and (explicit_proof or state.missing_fields == ("verification",)):
            if len(proof_values) != 1:
                return self._reply(state, "clarify", "Please provide just one verification value.", "कृपया सत्यापन का केवल एक मान दें।", "Verification ke liye sirf ek value dein.", ids=("IDV-2",))
            result = gateway.verify(sid, proof_values[0])
            self._sync(sid, state)
            if not result.verified:
                return self._verification_reply(state, result)
            state.missing_fields = ()
            state.checks.append("Caller identity verified for active order")
            # Resume a specific initial request; generic help returns only status.
            response = self._answer(sid, state, text)
            return replace(response, tool_events=((begin_event,) if begin_event else ()) + (result.event,) + response.tool_events)
        if state.intent in {"coupon", "delivery_absence"}:
            return self._answer(sid, state, text)
        if state.intent == "serviceability":
            return self._answer(sid, state, text)
        if not intent and state.intent == "return" and re.search(r"how long|kitn.*(?:time|din)|कितन.*(?:समय|दिन)", text, re.I):
            state.missing_fields = ("intent",)
            return self._reply(state, "clarify", "Do you mean the return pickup time or the refund time?", "आप रिटर्न पिकअप का समय पूछ रहे हैं या रिफ़ंड का?", "Return pickup time pooch rahe hain ya refund time?")
        if state.missing_fields == ("refund_method",):
            cod = state.verified_facts.get("payment_method") == "cod"
            if re.search(r"wallet|वॉलेट", text, re.I):
                state.caller_fields["refund_method"] = "cod_wallet" if cod else "wallet"
            elif re.search(r"bank|बैंक", text, re.I):
                state.caller_fields["refund_method"] = "cod_bank" if cod else "netbanking"
            elif re.search(r"\bupi\b|यूपीआई", text, re.I):
                state.caller_fields["refund_method"] = "upi"
            elif re.search(r"card|कार्ड", text, re.I):
                state.caller_fields["refund_method"] = "card"
        if state.missing_fields in {("slot_start",), ("current_time",)}:
            field_name = state.missing_fields[0]
            value = self._time(text)
            if value:
                state.caller_fields[field_name] = value
        private_request = bool(re.search(r"\b(my|mine|mera|meri|mujhe|ise|this order|that order)\b|मेरा|मेरी|मेरे|इस.*ऑर्डर", text, re.I))
        explicit_general = bool(re.search(r"policy|in general|आम तौर|नीति|generally|\b(?:upi|card|wallet|electronics|fashion)\b.*(?:refund|return)|(?:refund|return).*(?:upi|card|wallet|electronics|fashion)", text, re.I))
        if explicit_general and not private_request and not orders:
            return self._policy(sid, state, text)
        if state.intent in {"refund", "return", "cancel", "tracking", "damage", "warranty", "membership", "serviceability"} and not state.active_order and not private_request and not orders:
            return self._policy(sid, state, text)
        general = bool(re.search(r"policy|in general|आम तौर|नीति|generally", text, re.I)) and not orders
        if general:
            return self._policy(sid, state, text)
        if state.intent == "repeat_contact" and not state.active_order:
            return self._escalate(sid, state, ["repeat_contact"])
        if state.intent in {"order", "tracking", "refund", "return", "cancel", "damage", "compensation", "express_credit", "repeat_contact"}:
            if not state.active_order:
                state.missing_fields = ("order_id",)
                return self._reply(state, "clarify", "What is the 8-digit order ID shown in My Orders?", "My Orders में दिखने वाला 8 अंकों का ऑर्डर नंबर क्या है?", "My Orders mein dikhne wala 8-digit order ID kya hai?", ids=("IDV-1", "IDV-2"), code="order_id_required")
            if not state.verified:
                gate = gateway.verification_state(sid)
                if gate["attempts"] >= 2:
                    return self._locked(state)
                state.missing_fields = ("verification",)
                return self._ask_proof(state)
        return self._answer(sid, state, text)

    @staticmethod
    def _reset_issue(state):
        state.missing_fields = ()
        state.selected_line = None
        state.caller_fields = {}
        state.issue_messages = []
        state.checks = []
        state.last_answer = ""
        state.evidence_ids = ()

    def _record_issue(self, sid, state, text):
        safe = self.operations.gateway.sanitize(sid, text)
        if safe not in state.issue_messages:
            state.issue_messages.append(safe)
        state.issue_messages = state.issue_messages[-8:]

    def _ask_proof(self, state, events=()):
        return self._reply(state, "clarify", "What are the last four digits of your registered phone number?",
            "आपके रजिस्टर्ड फ़ोन नंबर के आख़िरी चार अंक क्या हैं?", "Registered phone number ke last four digits kya hain?", ids=("IDV-2",), events=events)

    def _locked(self, state, events=()):
        state.missing_fields = ()
        return self._reply(state, "refuse", "I cannot share order details after two failed checks; please contact support from your registered phone. I can still help with general policies.",
            "दो असफल जाँचों के बाद ऑर्डर विवरण साझा नहीं कर सकता; रजिस्टर्ड फ़ोन से सपोर्ट से संपर्क करें। सामान्य नीतियों में मदद कर सकता हूँ।",
            "Do failed checks ke baad order details nahi bata sakta; registered phone se support contact karein. General policies mein help kar sakta hoon.", ids=("IDV-3",), events=events)

    def _verification_reply(self, state, result):
        if result.status == "locked":
            return self._locked(state, (result.event,))
        state.missing_fields = ("verification",)
        if result.status == "tool_error":
            return self._reply(state, "clarify", "The order lookup is unavailable; could you try again?",
                "ऑर्डर जाँच अभी उपलब्ध नहीं है; क्या आप फिर कोशिश करेंगे?", "Order lookup abhi unavailable hai; phir try karenge?", events=(result.event,))
        if result.status == "verification_failed":
            return self._reply(state, "clarify", "That check did not match; what is the delivery pincode for this order?",
                "सत्यापन मेल नहीं खाया; इस ऑर्डर का डिलीवरी पिनकोड क्या है?", "Verification match nahi hua; is order ka delivery pincode kya hai?", ids=("IDV-3",), events=(result.event,))
        return self._ask_proof(state, (result.event,))

    @staticmethod
    def _reason(intent):
        return {"human": "human_requested", "unauthorized": "unauthorized_activity", "legal": "legal_threat", "safety": "safety_risk"}[intent]

    def _escalate(self, sid, state, reasons, *, events=(), extra=""):
        reason = ", ".join(sorted(set(reasons)))
        check = "Escalation policy checked: " + reason
        if check not in state.checks:
            state.checks.append(check)
        event = self.operations.gateway.handover(sid,
            issue_key=state.intent or "support", issue=" | ".join(state.issue_messages) or "Support requested",
            intent=state.intent or "support", checked="; ".join(state.checks) or "No private order details accessed",
            reason=reason)
        state.tickets[f"{state.active_order}:{state.intent}:{reason}"] = event.to_dict()
        state.missing_fields = ()
        if event.status == "succeeded" and event.details.get("created") is True:
            en, hi, hinglish = "I've created a ticket; a human colleague will contact you.", "टिकट बन गया है; एक मानव सहयोगी आपसे संपर्क करेगा।", "Ticket ban gaya hai; human colleague aapse contact karega."
        else:
            en, hi, hinglish = "A human colleague is needed, but ticket creation failed; please retry or contact support from your registered phone.", "मानव सहयोगी की ज़रूरत है, पर टिकट नहीं बन पाया; फिर कोशिश करें या रजिस्टर्ड फ़ोन से सपोर्ट से संपर्क करें।", "Human colleague ki zaroorat hai, par ticket nahi bana; phir try karein ya registered phone se support contact karein."
        if "safety_risk" in reasons:
            en = "Stop using the product immediately. " + en
            hi = "उत्पाद का इस्तेमाल तुरंत बंद करें। " + hi
            hinglish = "Product use karna turant band karein. " + hinglish
        if extra:
            en += " " + extra
            hi += " " + extra
            hinglish += " " + extra
        return self._reply(state, "escalate", en, hi, hinglish, ids=("ESC-1", "ESC-2", "ESC-3", "DWM-5") if "safety_risk" in reasons else ("ESC-1", "ESC-2", "ESC-3", "DWM-3") if "high_value_damage" in reasons else ("ESC-1", "ESC-2", "ESC-3", "REF-6") if extra else ("ESC-1", "ESC-2", "ESC-3"), events=(*events, event))

    def _answer(self, sid, state, text):
        facts, intent = state.verified_facts, state.intent
        if facts and intent not in {"order", "coupon", "delivery_absence"}:
            prior = int(facts.get("contacts_last_7_days") or 0)
            reasons = self.operations.calculator.escalation_reasons(prior_contacts=prior).values["reasons"]
            if reasons:
                state.checks.append("Verified earlier contacts: " + str(prior))
                return self._escalate(sid, state, reasons)
        if intent == "repeat_contact":
            return self._escalate(sid, state, ["repeat_contact"])
        if intent == "serviceability":
            pins = digit_values(text, 6)
            if len(pins) != 1:
                state.missing_fields = ("public_pincode",)
                return self._reply(state, "clarify", "Which delivery pincode would you like to check?", "आप किस डिलीवरी पिनकोड की जाँच चाहते हैं?", "Kaunsa delivery pincode check karna hai?")
            calculation = self.operations.pincode(pins[0])
            row = calculation.values["availability"]
            if row is None:
                return self._reply(state, "answer", "That pincode is absent from the supplied table; I cannot confirm availability.", "यह पिनकोड दी गई तालिका में नहीं है; सेवा की पुष्टि नहीं कर सकता।", "Yeh pincode supplied table mein nahi hai; availability confirm nahi kar sakta.", ids=("DEL-5",))
            self._sources.update({e.evidence_id: e for e in calculation.source_evidence})
            yes = [str(row[key]).lower() == "true" for key in ("serviceable", "cod_available", "express_available")]
            return self._reply(state, "answer", f"Delivery: {'available' if yes[0] else 'unavailable'}; cash on delivery: {'available' if yes[1] else 'unavailable'}; Express: {'available' if yes[2] else 'unavailable'}.",
                f"डिलीवरी: {'उपलब्ध' if yes[0] else 'अनुपलब्ध'}; कैश ऑन डिलीवरी: {'उपलब्ध' if yes[1] else 'अनुपलब्ध'}; एक्सप्रेस: {'उपलब्ध' if yes[2] else 'अनुपलब्ध'}।",
                f"Delivery {'available' if yes[0] else 'unavailable'} hai; COD {'available' if yes[1] else 'unavailable'} hai; Express {'available' if yes[2] else 'unavailable'} hai.", ids=calculation.evidence_ids)
        if intent in {"order", "tracking"} and facts:
            if intent == "order" and re.search(r"how (?:long|much)|kitn|कितन", text, re.I):
                state.missing_fields = ("intent",)
                return self._reply(state, "clarify", "Do you mean the delivery time or the refund time?", "आप डिलीवरी का समय पूछ रहे हैं या रिफ़ंड का?", "Delivery time pooch rahe hain ya refund time?")
            if re.search(r"ship.*(?:when|kab)|when.*ship|kab ship|कब.*(?:शिप|भेज)", text, re.I):
                shipped = self._date(facts.get("shipped_on"))
                return self._reply(state, "answer", f"It shipped on {shipped}." if facts.get("shipped_on") else "No shipping date is recorded yet.",
                    f"यह {shipped} को भेजा गया था।" if facts.get("shipped_on") else "अभी भेजने की तारीख दर्ज नहीं है।",
                    f"Order {shipped} ko ship hua tha." if facts.get("shipped_on") else "Shipping date abhi record nahi hai.", ids=("TRK-1",))
            status = str(facts.get("status", "unknown")).replace("_", " ")
            promised = self._date(facts.get("promised_date"))
            return self._reply(state, "answer", f"Your order status is {status}; the recorded promised date is {promised}.",
                f"ऑर्डर की स्थिति {status} है; दर्ज वादा की गई तारीख {promised} है।", f"Order status {status} hai; promised date {promised} hai.", ids=("TRK-1", "DEL-1"))
        if intent == "coupon":
            # Use the original source for open policy generation when enabled.
            return self._policy(sid, state, text, ids=("CPN-1",))
        if intent == "delivery_absence":
            return self._policy(sid, state, text, ids=("DEL-3",))
        if intent == "refund" and facts:
            return self._refund(sid, state)
        if intent == "return" and facts:
            item = self._item(state, text)
            if item is None:
                state.missing_fields = ("item",)
                return self._reply(state, "clarify", "Which item would you like to return?", "आप कौन सा सामान वापस करना चाहते हैं?", "Kaunsa item return karna hai?")
            result = self.operations.return_eligibility(sid, str(item["line_no"]))
            state.checks.append("Return window and category conditions checked")
            v = result.values
            state.missing_fields = ()
            if v.get("needs_delivery"):
                return self._reply(state, "answer", "The order has not been delivered; a return window cannot be confirmed yet.", "ऑर्डर अभी डिलीवर नहीं हुआ है; रिटर्न की अवधि अभी तय नहीं कर सकते।", ids=result.evidence_ids)
            if v.get("within_window"):
                deadline = self._date(v["deadline"])
                fee = self.operations.calculator.rules.pickup_fee if item["return_category"] == "fashion" and not facts["is_plus"] else Decimal(0)
                return self._reply(state, "answer", f"The return window ends on {deadline}, subject to the item's condition and original packaging. A change-of-mind pickup fee of {fee} rupees applies.",
                    f"रिटर्न की अंतिम तारीख {deadline} है; सामान की स्थिति और मूल पैकेजिंग की शर्तें लागू हैं। मन बदलने पर पिकअप शुल्क {fee} रुपये है।",
                    f"Return deadline {deadline} hai; item condition aur original packaging ki conditions apply hoti hain. Change-of-mind pickup fee {fee} rupees hai.", ids=(*result.evidence_ids, "RET-4", "RET-6"))
            return self._reply(state, "answer", "A change-of-mind return is unavailable under the recorded category and delivery dates; damage and warranty have separate conditions.",
                "दर्ज श्रेणी और डिलीवरी तारीख के अनुसार मन बदलने पर रिटर्न उपलब्ध नहीं है; खराब सामान और वारंटी की अलग शर्तें हैं।", ids=(*result.evidence_ids, "RET-5", "DWM-4"))
        if intent == "cancel" and facts:
            result = self.operations.calculator.cancellation(status=facts["status"], grocery=any(i["return_category"] == "grocery" for i in facts["items"]),
                slot_start=state.caller_fields.get("slot_start"), at=state.caller_fields.get("current_time"))
            state.checks.append("Cancellation status and grocery cutoff checked")
            if result.values.get("needs_clarification"):
                state.missing_fields = (result.values["missing"],)
                if result.values["missing"] == "slot_start":
                    return self._reply(state, "clarify", "What delivery slot start time is shown in My Orders?", "My Orders में डिलीवरी स्लॉट शुरू होने का समय क्या है?", "My Orders mein delivery slot ka start time kya hai?", ids=("CAN-3",))
                return self._reply(state, "clarify", "What is the current local time for this cancellation request?", "इस कैंसिल अनुरोध का वर्तमान स्थानीय समय क्या है?", "Is cancellation request ka current local time kya hai?", ids=("CAN-3",))
            state.missing_fields = ()
            if result.values.get("can_customer_cancel"):
                return self._reply(state, "answer", "You can cancel free in My Orders in the app before shipping; the refund is initiated on cancellation.",
                    "शिप होने से पहले ऐप के My Orders में मुफ़्त कैंसिल कर सकते हैं; कैंसिल होने पर रिफ़ंड शुरू होता है।", "Ship hone se pehle app ke My Orders mein free cancel kar sakte hain; cancellation par refund initiate hota hai.", ids=("CAN-1",))
            if facts["status"] in {"placed", "confirmed", "processing", "packed"}:
                return self._reply(state, "refuse", "The grocery cancellation cutoff has passed; cancellation is unavailable for this slot.", "इस स्लॉट के लिए ग्रॉसरी कैंसिल करने का समय बीत चुका है।", "Is grocery slot ka cancellation cutoff pass ho chuka hai.", ids=("CAN-3",))
            if facts["status"] not in {"shipped", "out_for_delivery", "delivery_attempted"}:
                return self._reply(state, "refuse", "Cancellation is unavailable for the recorded order status; I can help with any applicable return or refund policy.", "दर्ज ऑर्डर स्थिति में कैंसिल उपलब्ध नहीं है; लागू रिटर्न या रिफ़ंड नीति में मदद कर सकता हूँ।", ids=("CAN-1", "CAN-2"))
            return self._reply(state, "refuse", "A shipped order cannot be cancelled; you can refuse it at the door. For prepaid orders, the refund starts after warehouse return, usually 7 to 10 calendar days after the last delivery attempt.",
                "शिप किया ऑर्डर कैंसिल नहीं हो सकता; दरवाज़े पर लेने से मना कर सकते हैं। प्रीपेड रिफ़ंड गोदाम वापसी पर शुरू होता है, आम तौर पर आख़िरी डिलीवरी प्रयास के 7 से 10 कैलेंडर दिनों बाद।",
                "Shipped order cancel nahi ho sakta; door par lene se mana kar sakte hain. Prepaid refund warehouse return par start hota hai, usually last delivery attempt ke 7 se 10 calendar days baad.", ids=("CAN-2", "REF-1"))
        if intent == "damage" and facts:
            item = self._item(state, text, damage=True)
            if item is None:
                state.missing_fields = ("item",)
                return self._reply(state, "clarify", "Which item is damaged, defective or wrong?", "कौन सा सामान टूटा, खराब या गलत है?", "Kaunsa item damaged, defective ya wrong hai?")
            reasons = self.operations.calculator.escalation_reasons(damaged=True, item_value=Decimal(str(item["unit_price_inr"]))).values["reasons"]
            if reasons:
                state.checks.append("Verified affected item value and high-value damage threshold")
                return self._escalate(sid, state, reasons)
            return self._policy(sid, state, text, ids=("DWM-1", "DWM-2", "DWM-4"))
        if intent == "compensation":
            return self._reply(state, "refuse", "I'm sorry about the delay; I cannot offer a voucher or compensation beyond the delivery policy.",
                "देरी के लिए माफ़ कीजिए; डिलीवरी नीति से बाहर वाउचर या मुआवज़ा नहीं दे सकता।", "Delay ke liye maaf kijiye; policy ke bahar voucher ya compensation nahi de sakta.", ids=("DEL-4", "ESC-4"))
        if intent == "express_credit" and facts:
            amount, on = facts.get("express_late_credit_inr"), facts.get("express_late_credit_on")
            if amount and on:
                stamp = self._date(on)
                return self._reply(state, "answer", f"The recorded late Express fee credit is {amount} rupees to Wallet on {stamp}.",
                    f"वॉलेट में देर से आए एक्सप्रेस शुल्क का {amount} रुपये का क्रेडिट {stamp} को दर्ज है।", f"Wallet mein late Express fee ka {amount} rupees credit {stamp} ko recorded hai.", ids=("DEL-4",))
            return self._policy(sid, state, text, ids=("DEL-4",))
        if not intent:
            if self.rag:
                return self._policy(sid, state, text)
            return self._input_guidance(state)
        return self._policy(sid, state, text)

    @staticmethod
    def _date(value):
        if not value:
            return "not recorded"
        day = date.fromisoformat(str(value)[:10])
        return f"{day.day} {day.strftime('%B %Y')}"

    def _item(self, state, text, *, damage=False):
        items = state.verified_facts["items"]
        normalized = text.casefold()
        matches = [i for i in items if str(i["sku"]).casefold() in normalized or i["title"].casefold() in normalized
                   or i["title"].split(",")[0].casefold() in normalized]
        line = re.search(r"(?:line|item)\s*(\d+)\b", normalize_digits(text), re.I)
        ordinal = re.search(r'\b(first|second|third) item\b|(?:पहला|पहली|दूसरा|दूसरी) (?:सामान|आइटम)', text,re.I)
        if ordinal:
            index = {'first':0,'second':1,'third':2}.get((ordinal[1] or '').lower(),1 if 'दूस' in ordinal[0] else 0)
            matches = [items[index]] if index < len(items) else []
        if line:
            matches = [i for i in items if str(i["line_no"]) == line[1]]
        if not matches:
            # Only use an explicit recorded damaged line, never the highest value item.
            sku = state.verified_facts.get("damage_sku") if damage else None
            matches = [i for i in items if i["sku"] == sku] if sku else []
        if not matches and state.selected_line:
            matches = [i for i in items if str(i["line_no"]) == state.selected_line]
        if not matches and len(items) == 1:
            matches = items
        if len(matches) == 1:
            state.selected_line = str(matches[0]["line_no"])
            return matches[0]
        return None

    def _refund(self, sid, state):
        f = state.verified_facts
        kind = "cancellation" if f["status"] == "cancelled" else "rto" if f["status"] == "returned_to_origin" else "return"
        result = self.operations.refund_status(sid, kind)
        v = result.values
        state.checks.append("Refund initiation, settlement deadline and RRN checked")
        if v.get("initiated") and v.get("needs_initiation_or_method"):
            method = state.caller_fields.get("refund_method")
            if not method:
                state.missing_fields = ("refund_method",)
                return self._reply(state, "clarify", "Which refund payment method was confirmed?", "रिफ़ंड के किस भुगतान माध्यम की पुष्टि हुई थी?", "Refund ka kaunsa payment method confirm hua tha?", ids=("REF-2", "REF-3"))
            v.update(self.operations.calculator.refund_timeline(method, date.fromisoformat(f[kind + "_refund_initiated_on"])).values)
            state.missing_fields = ()
        if v.get("completed_on"):
            stamp = self._date(v["completed_on"])
            return self._reply(state, "answer", f"The refund is recorded as completed on {stamp}.", f"रिफ़ंड {stamp} को पूरा हुआ दर्ज है।", f"Refund {stamp} ko completed recorded hai.", ids=result.evidence_ids)
        if v.get("requires_escalation"):
            return self._escalate(sid, state, ["overdue_refund"], extra=f"Refund reference number (RRN): {v['rrn']}." if v.get("rrn") else "")
        if v.get("initiated"):
            latest = self._date(v.get("latest_date"))
            rrn = f" RRN: {v['rrn']}." if v.get("rrn") else ""
            return self._reply(state, "answer", f"The refund has been initiated; the maximum settlement date is {latest}." + rrn if v.get("latest_date") else "The refund has been initiated; Wallet settlement is within two hours, but the exact initiation time is not recorded." + rrn,
                f"रिफ़ंड शुरू हो चुका है; अधिकतम निपटान तारीख {latest} है।" + rrn if v.get("latest_date") else "रिफ़ंड शुरू है; वॉलेट में दो घंटे के भीतर आता है, पर शुरू होने का सही समय दर्ज नहीं है।" + rrn,
                f"Refund initiate ho chuka hai; maximum settlement date {latest} hai." + rrn if v.get("latest_date") else "Refund initiated hai; Wallet mein do ghante ke andar aata hai, exact initiation time recorded nahi hai." + rrn, ids=result.evidence_ids)
        if f.get("return_pickup_on") and not f.get("return_qc_passed_on") and not f["is_plus"]:
            return self._reply(state, "answer", "Pickup is recorded, but quality check (QC) is pending; the refund starts within two business days of pickup once QC passes.",
                "पिकअप दर्ज है, पर गुणवत्ता जाँच (QC) बाकी है; जाँच पास होने पर पिकअप के दो कार्यदिवसों के भीतर रिफ़ंड शुरू होता है।",
                "Pickup recorded hai, quality check (QC) pending hai; QC pass hone par pickup ke do business days mein refund initiate hota hai.", ids=("RET-7", "REF-1"))
        method = f.get(kind + "_refund_method") or state.caller_fields.get("refund_method") or f.get("payment_method")
        if method == "cod":
            state.missing_fields = ("refund_method",)
            return self._reply(state, "clarify", "Would you prefer the COD refund to Wallet or a verified bank account?", "COD रिफ़ंड वॉलेट में चाहिए या सत्यापित बैंक खाते में?", "COD refund Wallet mein chahiye ya verified bank account mein?", ids=("REF-3",))
        timeline = self.operations.calculator.refund_timeline(method, None).values
        state.missing_fields = ()
        window = f"{timeline['minimum']} to {timeline['maximum']} business days" if timeline["unit"] == "business_days" else f"instant, within {timeline['maximum']} hours"
        initiation = "at pickup for Plus members" if f["is_plus"] else "within two business days of pickup once QC passes"
        return self._reply(state, "answer", f"Refund initiation is {initiation}; after initiation, {method} settlement takes {window}.",
            f"रिफ़ंड {'Plus सदस्यों के लिए पिकअप पर' if f['is_plus'] else 'QC पास होने पर पिकअप के दो कार्यदिवसों के भीतर'} शुरू होता है; उसके बाद {method} में {'तुरंत, दो घंटे के भीतर' if timeline['unit'] == 'hours' else str(timeline['minimum']) + ' से ' + str(timeline['maximum']) + ' कार्यदिवस'} लगते हैं।",
            f"Refund {'Plus ke liye pickup par' if f['is_plus'] else 'QC pass hone par pickup ke do business days mein'} initiate hota hai; uske baad {method} mein {window} lagte hain.", ids=("RET-7", "REF-1", "REF-2", "PLS-2"))

    def _policy(self, sid, state, text, *, ids=()):
        state.missing_fields = ()
        if self.rag:
            context = {"introduced": state.introduced, "language": state.language, "intent": state.intent,
                       "missing_fields": state.missing_fields, "last_answer": state.last_answer,
                       "relevant_evidence_ids": state.evidence_ids, "customer_facts": state.issue_messages}
            safe_text = self.operations.gateway.sanitize(sid, text)
            response = self.rag.respond(CustomerInput(safe_text, sid), previous_topic=state.intent,
                                        conversation_state=self.operations.gateway.sanitize(sid, context))
            if response.decision == Decision.CLARIFY:
                state.missing_fields = ("policy_detail",)
            if response.decision == Decision.ESCALATE:
                return self._escalate(sid, state, ["policy_requires_human_review"])
            return response
        if state.intent == 'refund':
            method = next((name for name,pattern in (('upi',r'upi|यूपीआई'),('wallet',r'wallet|वॉलेट'),('card',r'card|कार्ड'),('netbanking',r'net.?bank|नेट.?बैंक')) if re.search(pattern,text,re.I)),None)
            if method:
                timeline=self.operations.calculator.refund_timeline(method,None).values
                span=(f"within {timeline['maximum']} hours" if timeline['unit']=='hours' else f"{timeline['minimum']} to {timeline['maximum']} business days")
                hindi=(f"{timeline['maximum']} घंटे के भीतर" if timeline['unit']=='hours' else f"{timeline['minimum']} से {timeline['maximum']} कार्यदिवसों में")
                settlement=f'{method} refunds settle {span}' if timeline['unit']=='hours' else f'{method} refunds settle in {span}'
                return self._reply(state,'answer',f"After initiation, {settlement}. The initiation step depends on the cancellation or return process.",
                    f"रिफ़ंड शुरू होने के बाद {method} रिफ़ंड {hindi} आ जाता है। रिफ़ंड शुरू होने का चरण कैंसिल या रिटर्न प्रक्रिया पर निर्भर है।",
                    f"Refund initiate hone ke baad {method} mein {span} lagte hain. Initiation cancellation ya return process par depend karta hai.",ids=('REF-1','REF-2','KB-00.3'))
        if not ids:
            ids = {"refund": ("REF-1", "REF-2", "REF-3"), "return": ("RET-2", "RET-4"),
                   "cancel": ("CAN-1", "CAN-2", "CAN-3"), "tracking": ("DEL-1",),
                   "damage": ("DWM-1", "DWM-2", "DWM-3"), "warranty": ("WAR-1",),
                   "membership": ("PLS-2",), "serviceability": ("DEL-5",)}.get(state.intent, ())
            ids = tuple(i for i in ids if i in self._sources)
        # Explicit offline route inspection: preserve authoritative source text,
        # without pretending this extract is a measured multilingual LLM answer.
        if ids == ("CPN-1",):
            return self._reply(state, "answer", "Only one coupon can be used per order.", "एक ऑर्डर में केवल एक कूपन इस्तेमाल कर सकते हैं।", "Ek order par sirf ek coupon use kar sakte hain.", ids=ids)
        if ids == ("DEL-3",):
            return self._reply(state, "answer", "The courier makes up to 3 delivery attempts; after a failed attempt, use the SMS to reschedule. After the third failure the parcel returns to the warehouse.",
                "कूरियर अधिकतम 3 डिलीवरी प्रयास करता है; असफल प्रयास के बाद SMS से नया समय चुनें। तीसरी असफलता पर पार्सल गोदाम वापस जाता है।",
                "Courier up to 3 delivery attempts karta hai; failed attempt ke baad SMS se reschedule karein. Third failure par parcel warehouse wapas jata hai.", ids=ids)
        if ids:
            excerpt = "\n".join(self._sources[i].text for i in ids)
            return self._reply(state, "answer", excerpt, excerpt, excerpt, ids=ids, code="offline_source_extract")
        return self._reply(state, "answer", "I don't have a grounded answer for that question in offline mode.",
            "ऑफ़लाइन मोड में इस सवाल का प्रमाणित जवाब उपलब्ध नहीं है।", "Offline mode mein is sawaal ka grounded answer available nahi hai.", code="offline_unsupported")

    def _time(self, text):
        """Caller-supplied time only; never substitutes the machine clock."""
        normalized = normalize_digits(text).strip()
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", normalized):
                return datetime.fromisoformat(normalized)
            match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", normalized, re.I)
            if match and 1 <= int(match[1]) <= 12:
                hour = int(match[1]) % 12 + (12 if match[3].lower() == "pm" else 0)
                return datetime.combine(self.operations.calculator.today, datetime.min.time()).replace(hour=hour, minute=int(match[2] or 0))
        except ValueError:
            pass
        return None
