"""Local retrieval, privacy and policy boundary tests. No generation API calls."""

import csv
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.agent import AgentOutputError
from src.config import ConfigurationError, PROJECT_ROOT, Settings
from src.dataset import PublicKnowledge
from src.llm import ChatCompletionClient
from src.rag import RagAgent, numeric_tokens, RAG_RESPONSE_SCHEMA
from src.retrieval.index import KnowledgeIndex, chunks, source_fingerprint
from src.rules.calculations import PolicyCalculator
from src.rules.policies import PolicyRules
from src.schemas import CustomerInput, Evidence, GenerationTrace
from src.tools.gateway import ToolGateway, VerificationError
from src.tools.operations import SupportOperations

ROOT = PROJECT_ROOT / "Dataset"


class FakeEmbedder:
    name = "local-test-double"

    def passages(self, texts):
        return np.array([[1., 0.] for text in texts])

    def queries(self, texts):
        return np.array([[1., 0.] for text in texts])


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = PublicKnowledge.load(ROOT)

    def test_index_contains_public_sources_only_and_preserves_full_sections(self):
        evidence = chunks(self.knowledge)
        self.assertEqual(len([e for e in evidence if e.source == "data/products.csv"]), 70)
        self.assertTrue(all(e.source.startswith("kb/") or e.source == "data/products.csv" for e in evidence))
        ret = next(e for e in evidence if e.evidence_id == "RET-2")
        self.assertIn("| Electronics and accessories | 7 days |", ret.text)
        self.assertEqual(ret.metadata["version"], "4.0")

    def test_current_policy_outweighs_conflicting_faq_at_similar_relevance(self):
        policy = Evidence("P-1", "kb/policy", "refunds", {"doc_type": "policy", "status": "current", "effective_from": "2026-01-01"})
        faq = Evidence("F-1", "kb/faq", "refunds", {"doc_type": "faq", "status": "current", "effective_from": "2026-01-01"})
        index = KnowledgeIndex((policy, faq), np.array([[0.99, .1], [1, 0]]), FakeEmbedder(), "test")
        self.assertEqual(index.search("refunds", k=1).candidates[0].evidence.evidence_id, "P-1")

    def test_historical_policy_requires_current_successor_and_known_delivery_date(self):
        evidence = tuple(e for e in chunks(self.knowledge) if e.evidence_id in {"RET-3", "RETv3-2"})
        index = KnowledgeIndex(evidence, FakeEmbedder().passages(evidence), FakeEmbedder(), "test")
        self.assertNotIn("RETv3-2", [h.evidence.evidence_id for h in index.search("electronics", k=2).candidates])
        result = index.search("electronics", k=2, delivered_on=date(2026, 7, 31))
        self.assertIn("RETv3-2", [h.evidence.evidence_id for h in result.candidates])
        self.assertIn("RET-3", [e.evidence_id for e in result.context])

    def test_exact_sku_and_stale_index_handling(self):
        index = KnowledgeIndex.build(ROOT, FakeEmbedder())
        found = index.search("Warranty on ELE-0012", k=3)
        self.assertIn("products.csv:ELE-0012", [h.evidence.evidence_id for h in found.candidates])
        with TemporaryDirectory() as folder:
            path = Path(folder)
            index.save(path)
            loaded = KnowledgeIndex.load(path, FakeEmbedder(), ROOT)
            self.assertEqual(loaded.fingerprint, source_fingerprint(ROOT))
            metadata = json.loads((path / "metadata.json").read_text())
            metadata["fingerprint"] = "outdated"
            (path / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "sources changed"):
                KnowledgeIndex.load(path, FakeEmbedder(), ROOT)

    def test_spoken_public_sku_ambiguity_is_marked_and_product_names_resolve(self):
        index = KnowledgeIndex.build(ROOT, FakeEmbedder())
        spoken = index.search("sku ele zero zero one to warranty", k=3)
        product = next(e for e in spoken.context if e.evidence_id == "products.csv:ELE-0012")
        self.assertEqual(product.metadata["uncertain_asr_candidate"], "true")
        named = index.search("Can I return the Rangoli Cotton Straight Kurta?", k=3)
        self.assertIn("products.csv:FSH-0026", [h.evidence.evidence_id for h in named.candidates])


class GroundingTests(unittest.TestCase):
    def setUp(self):
        evidence = next(e for e in chunks(PublicKnowledge.load(ROOT)) if e.evidence_id == "REF-2")
        self.index = KnowledgeIndex((evidence,), np.array([[1., 0.]]), FakeEmbedder(), "test")

    def response(self, text="UPI takes 1 to 3 business days.", ids=None):
        ids = ["REF-2"] if ids is None else ids
        return {"decision": "answer", "reply": text, "language": "en", "supported": True,
                "evidence_ids": ids}

    def test_real_source_text_and_ids_reach_prompt_but_labels_do_not(self):
        client = Mock()
        client.complete.return_value = GenerationTrace(json.dumps(self.response()), "test-double")
        response = RagAgent(client, self.index).respond(CustomerInput("UPI refund time?", "one"))
        self.assertEqual(response.evidence[0].evidence_id, "REF-2")
        prompt = client.complete.call_args.args[0][1]["content"]
        self.assertIn("How long refunds take", prompt)
        self.assertNotIn("reference_answer", prompt)

    def test_fabricated_citation_and_unsupported_numeric_claim_are_rejected(self):
        for data in (self.response(ids=["FAKE-1"]), self.response("UPI takes 900 days.")):
            client = Mock()
            client.complete.return_value = GenerationTrace(json.dumps(data), "test-double")
            with self.assertRaises(AgentOutputError):
                RagAgent(client, self.index).respond(CustomerInput("refund", "one"))

    def test_unsupported_question_can_abstain_without_fabricated_evidence(self):
        client = Mock()
        client.complete.return_value = GenerationTrace(json.dumps({"decision": "answer", "reply": "I do not have information on that.",
            "language": "en", "supported": False, "evidence_ids": []}), "test-double")
        result = RagAgent(client, self.index).respond(CustomerInput("midnight delivery?", "one"))
        self.assertEqual(result.evidence, ())

    def test_equivalent_numeric_spellings_keep_evidence_checks_strict(self):
        self.assertEqual(numeric_tokens('01 October; ₹2.00; ०३'), {'1','2','3'})
        source=Evidence('P-1','kb/test','Orders placed 01 October to 15 October 2026 have a 30 day return window.',
            {'doc_type':'policy','status':'current','effective_from':'2026-01-01'})
        index=KnowledgeIndex((source,),np.array([[1.,0.]]),FakeEmbedder(),'test')
        client=Mock();client.complete.return_value=GenerationTrace(json.dumps(self.response(
            'Orders placed 1 October to 15 October 2026 have a 30 day return window.',ids=['P-1'])),'test')
        self.assertEqual(RagAgent(client,index).respond(CustomerInput('return window?','s')).decision.value,'answer')
        schema=client.complete.call_args.kwargs['response_schema']
        self.assertEqual(schema['properties']['evidence_ids']['items']['enum'],['P-1'])
        self.assertEqual(schema['properties']['language']['enum'],['en'])
        self.assertNotIn('claims',schema['properties'])

    def test_caller_date_echo_does_not_authorize_an_invented_policy_duration(self):
        for text,valid in [('Ordered on 17 October 2026; UPI takes 1 to 3 business days.',True),
                           ('Ordered on 17 October 2026; UPI takes 17 days.',False)]:
            client=Mock();client.complete.return_value=GenerationTrace(json.dumps(self.response(text)),'test')
            call=lambda:RagAgent(client,self.index).respond(CustomerInput('Ordered on 17 October 2026. UPI timeline?','s'))
            if valid:self.assertEqual(call().reply,text)
            else:
                with self.assertRaises(AgentOutputError):call()

    def test_warranty_completion_uses_unseen_public_catalog_values_and_preserves_raw_model_text(self):
        for months in (18,24):
            source=Evidence('products.csv:ELE-0999','data/products.csv',f'sku: ELE-0999\ntitle: Sample Phone\nwarranty_months: {months}',
                {'doc_type':'catalog','status':'current','effective_from':'2026-01-01','sku':'ELE-0999','title':'Sample Phone'})
            index=KnowledgeIndex((source,),np.array([[1.,0.]]),FakeEmbedder(),'test')
            raw=json.dumps(self.response('The product has warranty coverage.',ids=[source.evidence_id]))
            client=Mock();client.complete.return_value=GenerationTrace(raw,'test')
            result=RagAgent(client,index).respond(CustomerInput('Warranty on ELE-0999?','s'))
            self.assertIn(f'{months}-month warranty',result.reply)
            self.assertEqual(result.generation.raw_output,raw)
            self.assertEqual(result.evidence[0].evidence_id,source.evidence_id)

    def test_same_number_with_wrong_warranty_unit_is_rejected(self):
        from src.rules.public_facts import complete_catalog_warranty
        source=Evidence('products.csv:TEST-0999','data/products.csv','warranty_months: 18',{'doc_type':'catalog'})
        with self.assertRaisesRegex(ValueError,'contradicts catalog'):
            complete_catalog_warranty('Warranty lasts 18 years.','warranty',[source],'en')

    def test_public_refund_dates_use_actual_policy_calendar_without_order_access(self):
        knowledge=PublicKnowledge.load(ROOT)
        index=KnowledgeIndex.build(ROOT,FakeEmbedder())
        client=Mock();reply='The latest settlement date is 27 October 2026, conditional on initiation on 17 October 2026.'
        client.complete.return_value=GenerationTrace(json.dumps(self.response(reply)), 'test')
        response=RagAgent(client,index,knowledge=knowledge).respond(CustomerInput(
            'My card refund was started on 17 October 2026. By which date?', 's'))
        payload=json.loads(client.complete.call_args.args[0][1]['content'])
        self.assertEqual(payload['calculations'][0]['values']['latest_date'],'2026-10-27')
        self.assertIsNone(payload['verified_order_facts'])
        self.assertEqual(response.tool_events,())

    def test_dates_with_individually_supported_digits_cannot_override_calculation(self):
        knowledge=PublicKnowledge.load(ROOT);index=KnowledgeIndex.build(ROOT,FakeEmbedder())
        client=Mock();client.complete.return_value=GenerationTrace(json.dumps(self.response(
            'The latest settlement date is 7 October 2026.')), 'test')
        with self.assertRaises(AgentOutputError):
            RagAgent(client,index,knowledge=knowledge).respond(CustomerInput(
                'My card refund was started on 17 October 2026.', 's'))

    def test_ambiguous_refund_dates_are_not_silently_assigned(self):
        from src.rules.public_facts import refund_calculations
        calculator=PolicyCalculator(PolicyRules.from_knowledge(PublicKnowledge.load(ROOT)))
        self.assertEqual(refund_calculations('My card refund started on 14 October; delivery was 10 October.',calculator),())
        self.assertEqual(refund_calculations('My card refund will be started on 25 October 2026.',calculator),())
        self.assertEqual(refund_calculations('Card or UPI refund started on 14 October.',calculator),())

    @patch("requests.post")
    def test_deferred_generation_never_touches_the_network(self, post):
        settings = Settings.load({"LLM_ENDPOINT": "https://example.com/chat/completions", "LLM_MODEL": "configured"})
        with self.assertRaisesRegex(ConfigurationError, "deferred"):
            ChatCompletionClient(settings)
        post.assert_not_called()


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "data/orders.csv").open() as handle:
            orders = list(csv.DictReader(handle))
        with (ROOT / "data/customers.csv").open() as handle:
            customers = {r["customer_id"]: r for r in csv.DictReader(handle)}
        cls.order = orders[0]
        cls.proof = customers[cls.order["customer_id"]]["phone"][-4:]

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name)
        self.gateway = ToolGateway(ROOT, self.runtime)

    def test_caller_verified_lookup_and_no_expected_proof_disclosure(self):
        pending = self.gateway.begin("one", self.order["order_id"])
        self.assertEqual(pending.status, "verification_required")
        with self.assertRaises(VerificationError):
            self.gateway.order_facts("one")
        result = self.gateway.verify("one", self.proof)
        self.assertTrue(result.verified)
        facts = self.gateway.order_facts("one")
        self.assertNotIn("delivery_pincode", facts)
        self.assertNotIn("phone_last4", facts)
        self.assertNotIn(self.proof, json.dumps(result.event.to_dict()))
        facts.clear()
        self.assertIn("items", self.gateway.order_facts("one"))

    def test_native_script_pincode_is_accepted_but_not_exposed(self):
        self.gateway.begin("one", self.order["order_id"])
        native = self.order["delivery_pincode"].translate(str.maketrans("0123456789", "०१२३४५६७८९"))
        self.assertTrue(self.gateway.verify("one", native).verified)
        sanitized = self.gateway.sanitize("one", "Proof was " + native)
        self.assertNotIn(native, sanitized)

    def test_backend_cannot_unlock_a_different_order(self):
        backend = Mock()
        backend.lookup_order.side_effect = [{"verification_required": True},
            {"found": True, "verified": True, "order_id": "99999999", "items": []}]
        gateway = ToolGateway(ROOT, self.runtime, backend=backend)
        gateway.begin("one", self.order["order_id"])
        self.assertFalse(gateway.verify("one", self.proof).verified)
        with self.assertRaises(VerificationError):
            gateway.order_facts("one")

    def test_two_failures_lock_and_order_switch_does_not_reset_attempts(self):
        order_id = self.order["order_id"]
        wrong = "0000" if self.proof != "0000" else "9999"
        self.gateway.begin("one", order_id)
        self.assertEqual(self.gateway.verify("one", wrong).attempts_remaining, 1)
        self.assertEqual(self.gateway.verify("one", wrong).status, "locked")
        self.gateway.begin("one", "99999999")
        self.assertEqual(self.gateway.begin("one", order_id).status, "locked")
        self.assertFalse(self.gateway.verify("one", self.proof).verified)
        with self.assertRaises(VerificationError):
            self.gateway.order_facts("one")

    def test_sessions_and_orders_are_isolated(self):
        self.gateway.begin("one", self.order["order_id"])
        self.gateway.verify("one", self.proof)
        self.gateway.begin("two", self.order["order_id"])
        with self.assertRaises(VerificationError):
            self.gateway.order_facts("two")
        self.gateway.begin("one", "99999999")
        with self.assertRaises(VerificationError):
            self.gateway.order_facts("one")

    def test_ticket_success_redaction_idempotency_and_restart(self):
        self.gateway.begin("one", self.order["order_id"])
        self.gateway.verify("one", self.proof)
        args = dict(issue_key="damage-1", issue=f"Damaged item; submitted digits {self.proof}; email user@example.com",
                    intent="damage", checked="Policy and verified order", reason="high_value_damage")
        first = self.gateway.handover("one", **args)
        second = self.gateway.handover("one", **args)
        self.assertEqual(first.details["ticket_id"], second.details["ticket_id"])
        restarted = ToolGateway(ROOT, self.runtime)
        restarted.begin("one", self.order["order_id"])
        third = restarted.handover("one", **args)
        self.assertEqual(first.details["ticket_id"], third.details["ticket_id"])
        with (self.runtime / "tickets_created.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertNotIn(self.proof, rows[0]["summary"])
        self.assertNotIn("user@example.com", rows[0]["summary"])
        self.assertTrue(first.details["created"])

    def test_tool_failure_is_not_an_identity_failure_and_private_text_is_sanitized(self):
        backend = Mock()
        backend.lookup_order.side_effect = [{"verification_required": True}, OSError("private backend failure")]
        gateway = ToolGateway(ROOT, self.runtime, backend=backend)
        gateway.begin("one", self.order["order_id"])
        result = gateway.verify("one", "1234")
        self.assertEqual(result.status, "tool_error")
        self.assertEqual(result.attempts_remaining, 2)
        value = gateway.sanitize("one", {"nested": ["Reach me at +91 51234 56789 or someone@example.com"]})
        self.assertNotIn("51234", json.dumps(value))
        self.assertNotIn("someone@", json.dumps(value))

    def test_failed_ticket_never_claims_success_and_can_retry(self):
        backend = Mock()
        backend.create_ticket.side_effect = [{"created": False}, {"created": True, "ticket_id": "test-ticket"}]
        gateway = ToolGateway(ROOT, self.runtime, backend=backend)
        args = dict(issue_key="one", issue="Needs a human colleague", intent="human", checked="No order provided", reason="human_requested")
        self.assertEqual(gateway.handover("one", **args).status, "failed")
        self.assertEqual(gateway.handover("one", **args).status, "succeeded")

    def test_operational_calculations_require_verification(self):
        operations = SupportOperations(ROOT, self.runtime, gateway=self.gateway)
        self.gateway.begin("one", self.order["order_id"])
        with self.assertRaises(VerificationError):
            operations.refund_status("one", "return")
        self.assertTrue(operations.product("ELE-0012").values["found"])
        self.assertIsNone(operations.pincode("000000").values["availability"])


class CalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator = PolicyCalculator(PolicyRules.from_knowledge(PublicKnowledge.load(ROOT)))

    def test_policy_updates_change_calculations_without_hardcoded_answers(self):
        knowledge = PublicKnowledge.load(ROOT)
        documents = []
        for doc in knowledge.documents:
            sections = []
            for section in doc.sections:
                text = section.text
                if section.evidence_id == "RET-2":
                    text = text.replace("| Electronics and accessories | 7 days |", "| Electronics and accessories | 9 days |")
                if section.evidence_id == "REF-2":
                    text = text.replace("| UPI | 1 to 3 business days |", "| UPI | 2 to 4 business days |")
                sections.append(replace(section, text=text))
            documents.append(replace(doc, sections=tuple(sections)))
        updated = PolicyCalculator(PolicyRules.from_knowledge(replace(knowledge, documents=tuple(documents))))
        self.assertEqual(updated.return_window("electronics", date(2026, 10, 13), date(2026, 10, 10)).values["window_days"], 9)
        self.assertEqual(updated.refund_timeline("upi", date(2026, 10, 16)).values["latest_date"], "2026-10-22")

    def test_business_days_exclude_sundays_holidays_but_include_saturdays(self):
        self.assertEqual(self.calculator.add_business_days(date(2026, 10, 16), 1), date(2026, 10, 17))
        self.assertEqual(self.calculator.add_business_days(date(2026, 10, 16), 2), date(2026, 10, 20))
        self.assertEqual(self.calculator.add_business_days(date(2026, 10, 16), 3), date(2026, 10, 21))

    def test_current_and_grandfathered_electronics_and_festive_fashion(self):
        current = self.calculator.return_window("electronics", date(2026, 10, 13), date(2026, 10, 10))
        self.assertTrue(current.values["within_window"])
        self.assertEqual(current.values["deadline"], "2026-10-20")
        old = self.calculator.return_window("electronics", date(2026, 7, 31), date(2026, 7, 28))
        self.assertEqual(old.values["window_days"], 10)
        sale = self.calculator.return_window("fashion", date(2026, 10, 16), date(2026, 10, 15))
        self.assertEqual(sale.values["window_days"], 30)
        regular = self.calculator.return_window("fashion", date(2026, 10, 17), date(2026, 10, 16))
        self.assertEqual(regular.values["window_days"], 15)

    def test_refund_methods_and_date_only_wallet_ambiguity(self):
        timeline = self.calculator.refund_timeline("upi", date(2026, 10, 16))
        self.assertEqual(timeline.values["latest_date"], "2026-10-21")
        wallet = self.calculator.refund_timeline("cod_wallet", date(2026, 10, 16))
        self.assertTrue(wallet.values["needs_initiation_time_for_exact_deadline"])
        self.assertIsNone(wallet.values["exact_deadline"])
        with self.assertRaises(ValueError):
            self.calculator.refund_timeline("cod", date(2026, 10, 16))

    def order(self):
        return {"items": [{"line_no": "1", "qty": 2, "unit_price_inr": 200, "return_category": "fashion"},
                          {"line_no": "2", "qty": 1, "unit_price_inr": 400, "return_category": "electronics"}],
                "subtotal_inr": 800, "coupon_discount_inr": 100, "coupon_min_order_inr": 700,
                "delivery_fee_inr": 40, "cod_fee_inr": 25, "status": "packed"}

    def test_partial_quantities_coupon_threshold_fee_waiver_and_full_order_fees(self):
        order = self.order()
        quote = self.calculator.refund_quote(order, {"1": 1}, is_plus=False, reason="change_of_mind")
        self.assertEqual(quote.values["estimate_inr"], 51)  # 200 - entire lost 100 coupon - 49 pickup
        plus = self.calculator.refund_quote(order, {"1": 1}, is_plus=True, reason="change_of_mind")
        self.assertEqual(plus.values["estimate_inr"], 100)
        full = self.calculator.refund_quote(order, {"1": 2, "2": 1}, is_plus=False, reason="damaged")
        self.assertEqual(full.values["estimate_inr"], 765)
        self.assertFalse(full.values["refund_issued"])
        with self.assertRaises(ValueError):
            self.calculator.refund_quote(order, {"1": 3}, is_plus=False, reason="damaged")

    def test_half_rupee_rounds_up_and_no_false_cancellation(self):
        order = self.order()
        order.update(subtotal_inr=400, coupon_discount_inr=1, coupon_min_order_inr=0)
        order["items"] = [{"line_no": "1", "qty": 2, "unit_price_inr": 200, "return_category": "electronics"}]
        result = self.calculator.refund_quote(order, {"1": 1}, is_plus=True, reason="change_of_mind")
        self.assertEqual(result.values["estimate_inr"], 200)
        result = self.calculator.cancellation(status="shipped")
        self.assertFalse(result.values["can_customer_cancel"])
        self.assertFalse(result.values["assistant_can_cancel"])

    def test_missing_grocery_slot_clarification_and_exact_cutoff(self):
        self.assertTrue(self.calculator.cancellation(status="packed", grocery=True).values["needs_clarification"])
        result = self.calculator.cancellation(status="packed", grocery=True,
            slot_start=datetime(2026, 10, 20, 16), at=datetime(2026, 10, 20, 14))
        self.assertTrue(result.values["can_customer_cancel"])
        result = self.calculator.cancellation(status="shipped", grocery=True,
            slot_start=datetime(2026, 10, 20, 16), at=datetime(2026, 10, 20, 13))
        self.assertFalse(result.values["can_customer_cancel"])

    def test_escalation_thresholds_are_strict_and_prior_contacts_exclude_current(self):
        self.assertEqual(self.calculator.escalation_reasons(prior_contacts=1, overdue_business_days=3,
            damaged=True, item_value=Decimal(10000)).values["reasons"], [])
        result = self.calculator.escalation_reasons(prior_contacts=2, overdue_business_days=4,
            damaged=True, item_value=Decimal(10001), human_requested=True)
        self.assertEqual(set(result.values["reasons"]), {"repeat_contact", "overdue_refund", "high_value_damage", "human_requested"})


if __name__ == "__main__":
    unittest.main()
