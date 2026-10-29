"""Phase 4 routing, isolation, recovery and evaluator leakage regressions."""

from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from src.config import PROJECT_ROOT
from src.conversation import ConversationAgent
from src.evaluation.conversations import replay_conversations
from src.schemas import AgentResponse, CustomerInput, Decision
from src.tools.gateway import ToolGateway
from src.tools.operations import SupportOperations

ROOT = PROJECT_ROOT / "Dataset"


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name)
        self.ops = SupportOperations(ROOT, self.runtime / "tools")
        self.agent = ConversationAgent(self.ops)

    def say(self, text, session="one"):
        return self.agent.respond(CustomerInput(text, session))

    def verify(self, order="18599253", proof="4265", session="one"):
        self.say(f"My order number is {order}", session)
        return self.say(f"Last four digits {proof}", session)

    def fake_order(self, **updates):
        self.verify()
        facts = self.ops.gateway.order_facts("one")
        facts.update(contacts_last_7_days=0)
        facts.update(updates)
        backend = Mock()
        backend.lookup_order.side_effect = lambda oid, verification_value=None: deepcopy(facts) if verification_value else {"verification_required": True}
        backend.create_ticket.return_value = {"created": True, "ticket_id": "test-ticket"}
        self.ops.gateway = ToolGateway(ROOT, self.runtime / "fake", backend=backend)
        self.agent.clear_session("one")
        self.verify()
        return backend

    def test_first_reply_introduction_once_and_one_question(self):
        first = self.say("Hello, I need help with an order")
        self.assertIn("AI", first.reply)
        self.assertEqual(first.reply.count("?"), 1)
        second = self.say("Order 1859 9253")
        self.assertNotIn("AI", second.reply)
        self.assertEqual(second.reply.count("?"), 1)
        self.assertEqual(second.tool_events[0].name, "lookup_order")
        self.assertFalse(self.agent.snapshot("one").verified)

    def test_interleaved_sessions_and_order_switch_clear_verified_facts(self):
        self.verify()
        self.say("My order is 79363550", "two")
        self.assertFalse(self.agent.snapshot("two").verified)
        self.assertTrue(self.agent.snapshot("one").verified)
        response = self.say("Another order 79363550")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertEqual(self.agent.snapshot("one").verified_facts, {})
        response = self.say("When was it shipped?")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertNotIn("October", response.reply)

    def test_unspecified_new_order_never_uses_old_order(self):
        self.verify()
        response = self.say("Please help with a different order")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertIsNone(self.agent.snapshot("one").active_order)
        self.assertFalse(self.ops.gateway.verification_state("one")["verified"])

    def test_two_failed_attempts_survive_order_switch_and_general_help_works(self):
        self.say("Order 18599253")
        self.assertEqual(self.say("Last four 0000").decision, Decision.CLARIFY)
        self.assertEqual(self.say("Last four 0000").decision, Decision.REFUSE)
        self.say("Order 79363550")
        self.assertEqual(self.say("Order 18599253").decision, Decision.REFUSE)
        self.assertEqual(self.say("Can I use two coupons?").decision, Decision.ANSWER)

    def test_native_script_verification_and_language_switch(self):
        self.say("ऑर्डर नंबर ७९३६ ३५५० है।")
        response = self.say("आख़िरी चार अंक २९४९ हैं।")
        self.assertTrue(self.agent.snapshot("one").verified)
        self.assertEqual(response.language, "hi")
        response = self.say("Please reply in English. Can I use two coupons?")
        self.assertEqual(response.language, "en")
        self.assertIn("one coupon", response.reply)
        self.assertNotIn("2949", repr(self.agent.snapshot("one").issue_messages))

    def test_return_followup_has_correct_item_deadline_and_payment_method(self):
        self.verify()
        returned = self.say("Can I return the Kaafi Running Sneakers?")
        self.assertIn("27 October 2026", returned.reply)
        self.assertEqual(self.agent.snapshot("one").selected_line, "1")
        refunded = self.say("How long will that refund take?")
        self.assertIn("wallet", refunded.reply)
        self.assertIn("2 hours", refunded.reply)
        self.assertNotIn("5 to 7", refunded.reply)
        self.assertIn("at pickup", refunded.reply)

    def test_topic_switch_discards_old_evidence_and_can_return_to_order(self):
        self.verify()
        self.say("Can I return the Kaafi Running Sneakers?")
        coupon = self.say("Different question: can I use two coupons?")
        self.assertEqual([e.evidence_id for e in coupon.evidence], ["CPN-1"])
        self.assertIsNone(self.agent.snapshot("one").selected_line)
        shipment = self.say("Going back to my order, when was it shipped?")
        self.assertEqual(shipment.decision, Decision.ANSWER)
        self.assertTrue(self.agent.snapshot("one").verified)
        self.assertEqual([e.evidence_id for e in shipment.evidence], ["TRK-1"])

    def test_switching_back_uses_only_verification_for_that_order(self):
        self.verify()
        self.say("Order 79363550")
        self.say("Last four 2949")
        response = self.say("Back to order 18599253")
        self.assertEqual(response.decision, Decision.ANSWER)
        self.assertEqual(self.agent.snapshot("one").active_order, "18599253")
        self.assertTrue(self.agent.snapshot("one").verified)
        self.say("Can I use two coupons?")
        self.assertEqual(self.say("Back to order 18599253").decision, Decision.ANSWER)
        self.assertEqual(self.agent.snapshot("one").intent, "order")

    def test_ambiguous_followup_and_multiple_orders_ask_one_question(self):
        self.verify()
        response = self.say("How long will that take?")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertEqual(response.reply.count("?"), 1)
        response = self.say("Compare order 18599253 and order 79363550")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertEqual(response.tool_events, ())

    def test_general_policy_does_not_require_identity(self):
        response = self.say("How long do refunds take in general?")
        self.assertEqual(response.decision, Decision.ANSWER)
        self.assertEqual(response.tool_events, ())
        self.assertIsNone(self.agent.snapshot("one").active_order)
        self.assertIn("REF-2", [e.evidence_id for e in response.evidence])

    def test_human_security_legal_safety_escalate_without_verification(self):
        for session, text in enumerate(("I want a human agent", "There is an unauthorized payment", "I will take legal action", "The battery is swelling")):
            response = self.say(text, str(session))
            self.assertEqual(response.decision, Decision.ESCALATE)
            self.assertEqual(response.tool_events[-1].name, "create_ticket")
            self.assertTrue(response.tool_events[-1].details["created"])
        self.assertIn("Stop using", response.reply)

    def test_high_value_damage_uses_affected_item_and_strict_threshold(self):
        for amount, expected in ((10000, Decision.ANSWER), (10001, Decision.ESCALATE)):
            self.fake_order(items=[{"line_no": "1", "sku": "TEST", "title": "Test Device", "unit_price_inr": amount, "qty": 1, "return_category": "electronics"}])
            response = self.say("My Test Device is damaged")
            self.assertEqual(response.decision, expected)

    def test_reported_high_value_damage_can_handover_without_order_disclosure(self):
        response = self.say("My device worth 15000 rupees is damaged")
        self.assertEqual(response.decision, Decision.ESCALATE)
        self.assertEqual([e.name for e in response.tool_events], ["create_ticket"])
        self.assertFalse(self.agent.snapshot("one").verified)

    def test_repeat_contact_counts_prior_contacts_and_needs_no_explicit_complaint(self):
        self.fake_order(contacts_last_7_days=2)
        response = self.say("Where is my refund?")
        self.assertEqual(response.decision, Decision.ESCALATE)
        backend = self.ops.gateway.backend
        self.assertIn("repeat_contact", backend.create_ticket.call_args.kwargs["reason"])

    def test_failed_ticket_is_honest_retryable_and_success_is_idempotent(self):
        backend = Mock()
        backend.create_ticket.side_effect = [{"created": False}, {"created": True, "ticket_id": "retry-ticket"}]
        self.ops.gateway = ToolGateway(ROOT, self.runtime / "errors", backend=backend)
        failed = self.say("I want a human agent")
        self.assertIn("failed", failed.reply)
        self.assertFalse(failed.tool_events[-1].details["created"])
        succeeded = self.say("I want a human agent")
        again = self.say("I want a human agent")
        self.assertTrue(succeeded.tool_events[-1].details["created"])
        self.assertEqual(again.tool_events[-1].details["ticket_id"], "retry-ticket")
        self.assertEqual(backend.create_ticket.call_count, 2)

    def test_handover_summary_has_context_checks_reason_but_no_proof(self):
        self.verify()
        self.say("I want a human agent; my email is caller@example.com")
        with (self.runtime / "tools/tickets_created.csv").open() as handle:
            row = next(csv.DictReader(handle))
        for part in ("Issue:", "Intent:", "Order: 18599253", "Checks:", "Escalation reason: human_requested"):
            self.assertIn(part, row["summary"])
        self.assertNotIn("4265", row["summary"])
        self.assertNotIn("caller@example.com", row["summary"])

    def test_tool_error_does_not_consume_identity_retry(self):
        backend = Mock()
        backend.lookup_order.side_effect = [{"verification_required": True}, OSError("private failure")]
        self.ops.gateway = ToolGateway(ROOT, self.runtime / "errors", backend=backend)
        self.say("My order is 18599253")
        response = self.say("Last four 4265")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertEqual(self.agent.snapshot("one").verification_attempts, 0)
        self.assertNotIn("private failure", response.reply)

    def test_cod_method_followup_and_grocery_missing_times(self):
        self.fake_order(payment_method="cod", return_refund_initiated_on=None, return_refund_method=None, return_pickup_on=None)
        self.assertEqual(self.say("When will my refund arrive?").decision, Decision.CLARIFY)
        response = self.say("Wallet please")
        self.assertEqual(response.decision, Decision.ANSWER)
        self.assertIn("2 hours", response.reply)
        self.fake_order(status="packed", items=[{"line_no": "1", "sku": "TEST", "title": "Food", "return_category": "grocery", "unit_price_inr": 100, "qty": 1}])
        self.assertEqual(self.say("Cancel my order").decision, Decision.CLARIFY)
        self.assertEqual(self.say("4 pm").decision, Decision.CLARIFY)
        self.assertEqual(self.agent.snapshot("one").missing_fields, ("current_time",))
        response = self.say("2 pm")
        self.assertEqual(response.decision, Decision.ANSWER)
        self.assertIn("My Orders", response.reply)
        self.assertNotIn("cancelled", response.reply)

    def test_snapshot_is_copy_and_clear_session_removes_verification(self):
        self.verify()
        snapshot = self.agent.snapshot("one")
        snapshot.verified_facts.clear()
        self.assertTrue(self.agent.snapshot("one").verified_facts)
        self.agent.clear_session("one")
        self.assertEqual(self.agent.snapshot("one").turns, 0)
        self.assertIsNone(self.ops.gateway.verification_state("one")["order_id"])

    def test_rag_context_is_relevant_and_proofs_never_reach_model(self):
        rag = Mock()
        rag.respond.return_value = AgentResponse(Decision.ANSWER, "Only one coupon.", language="en")
        self.agent.rag = rag
        self.verify()
        self.say("Last four 4265; what is the coupon policy?")
        args, kwargs = rag.respond.call_args
        self.assertNotIn("4265", str(args) + str(kwargs))
        self.assertNotIn("verified_order_facts", kwargs)
        self.assertEqual(kwargs["conversation_state"]["intent"], "coupon")
        self.assertNotIn("promised_date", str(kwargs))

    def test_generated_proof_leak_is_refused(self):
        self.verify()
        rag = Mock()
        rag.respond.return_value = AgentResponse(Decision.ANSWER, "Your proof is 4265.")
        self.agent.rag = rag
        response = self.say("What is the coupon policy?")
        self.assertEqual(response.decision, Decision.REFUSE)
        self.assertNotIn("4265", response.reply)

    def test_public_pincode_lookup_is_not_identity_and_unknown_stays_unknown(self):
        response = self.say("Is pincode 560034 serviceable?")
        self.assertEqual(response.decision, Decision.ANSWER)
        self.assertIn("Delivery: available", response.reply)
        self.assertIn("pincodes.csv:560034", [e.evidence_id for e in response.evidence])
        self.assertIsNone(self.agent.snapshot("one").active_order)
        self.assertIn("cannot confirm", self.say("Is pincode 000000 serviceable?").reply)

    def test_pincode_verification_resumes_original_request(self):
        self.say("My order number is 15911102")
        response = self.say("Pincode 440010 hai")
        self.assertEqual(response.decision, Decision.ANSWER)
        self.assertEqual(self.agent.snapshot("one").intent, "order")
        self.assertTrue(self.agent.snapshot("one").verified)
        self.assertIn("Order status", response.reply)
        self.assertNotIn("440010", json.dumps(response.to_dict()))

    def test_inline_order_and_proof_logs_both_lookups(self):
        response = self.say("My order is 18599253; last four digits 4265")
        self.assertTrue(self.agent.snapshot("one").verified)
        self.assertEqual([e.name for e in response.tool_events], ["lookup_order", "lookup_order"])
        self.assertNotIn("4265", json.dumps(response.to_dict()))

    def test_missing_initiated_refund_method_clarifies_instead_of_assuming_wallet(self):
        self.fake_order(return_refund_initiated_on="2026-10-16", return_refund_method=None, return_refund_completed_on=None)
        response = self.say("Where is my refund?")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertNotIn("Wallet", response.reply)
        response = self.say("UPI")
        self.assertEqual(response.decision, Decision.ANSWER)
        self.assertIn("21 October 2026", response.reply)

    def test_ambiguous_return_duration_does_not_invent_pickup_booking(self):
        self.verify()
        self.say("Can I return the Kaafi Running Sneakers?")
        response = self.say("How long will that take?")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertEqual(response.reply.count("?"), 1)
        self.assertEqual(response.tool_events, ())

    def test_full_replay_all_turns_with_redacted_artifacts(self):
        result = replay_conversations(ROOT, self.runtime / "replay", self.agent)
        self.assertEqual(result["processed"], 96)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["decision_matches"], 96)
        self.assertEqual(result["tool_call_matches"], 96)
        self.assertEqual(result["scenarios_passing_routing"], 23)
        self.assertEqual(result["fact_correctness"], "not_scored")
        records = [json.loads(l) for l in (self.runtime / "replay/results.jsonl").read_text().splitlines()]
        # Random ticket IDs and fractional timing measurements can coincidentally
        # contain four digits. Inspect customer-visible/model text for leakage.
        persisted_text = '\n'.join(r['response']['reply'] + '\n' + (r['response'].get('generation') or {}).get('raw_output','') for r in records)
        self.assertNotIn("4265", persisted_text)
        self.assertNotIn("expected_tool_calls", json.dumps(records))
        self.assertNotIn("customer_message", json.dumps(records))


if __name__ == "__main__":
    unittest.main()
