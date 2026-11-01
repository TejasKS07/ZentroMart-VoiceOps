"""Unclear-input repair and safe console observability regressions."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.chat import run_chat
from src.config import PROJECT_ROOT, Settings
from src.conversation import ConversationAgent
from src.llm import LLMError
from src.schemas import CustomerInput, Decision
from src.tools.operations import SupportOperations


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agent = ConversationAgent(SupportOperations(PROJECT_ROOT / "Dataset", Path(self.temp.name) / "tools"))

    def say(self, text, session="feedback"):
        return self.agent.respond(CustomerInput(text, session))

    def console(self, *messages, debug=False):
        output = StringIO()
        with patch("builtins.input", side_effect=messages), redirect_stdout(output):
            run_chat(self.agent, debug=debug)
        return output.getvalue()

    def test_reported_sequence_has_guidance_and_specific_length_errors(self):
        first = self.say("wjhat")
        second = self.say("what")
        self.assertIn("delivery", first.reply)
        self.assertIn("track my order", second.reply)
        self.assertNotEqual(first.reply, second.reply)
        self.assertEqual(first.diagnostics["code"], "input_not_understood")
        third = self.say("what")
        self.assertIn("human agent", third.reply)
        self.assertEqual(third.tool_events, ())
        self.assertEqual(self.say("new order").diagnostics["code"], "order_goal_ambiguous")
        with patch.object(self.agent.operations.gateway.backend, "lookup_order") as lookup:
            for text, size in (("1234", 4), ("123455", 6)):
                response = self.say(text)
                self.assertIn(f"{size} digits", response.reply)
                self.assertIn("exactly 8", response.reply)
                self.assertEqual(response.diagnostics["code"], "invalid_order_id")
                self.assertEqual(response.diagnostics["tools"], [])
                self.assertEqual(response.diagnostics["waiting_for"], ["order_id"])
            lookup.assert_not_called()
        self.assertEqual(self.agent.snapshot("feedback").verification_attempts, 0)
        accepted = self.say("1859 9253")
        self.assertEqual(accepted.diagnostics["code"], "verification_required")
        self.assertEqual(accepted.tool_events[0].name, "lookup_order")

    def test_new_order_does_not_pretend_it_can_buy_or_start_lookup(self):
        response = self.say("I want to place a new order")
        self.assertIn("isn't available", response.reply)
        self.assertEqual(response.diagnostics["code"], "order_placement_unavailable")
        self.assertEqual(response.tool_events, ())
        self.assertIsNone(self.agent.snapshot("feedback").active_order)
        self.say("new order", "other")
        response = self.say("what", "other")
        self.assertEqual(response.diagnostics["code"], "order_goal_ambiguous")
        response = self.say("existing order", "other")
        self.assertIn("8-digit", response.reply)
        self.assertEqual(response.diagnostics["waiting_for"], ["order_id"])

    def test_invalid_and_native_ids_do_not_get_guessed_or_padded(self):
        for invalid in ("123", "123456789", "AB12345678", "1234 5678XYZ", "12A345678"):
            self.say("track my order")
            with patch.object(self.agent.operations.gateway.backend, "lookup_order") as lookup:
                response = self.say(invalid)
                lookup.assert_not_called()
            self.assertEqual(response.diagnostics["code"], "invalid_order_id")
        response = self.say("१८५९-९२५३")
        self.assertEqual(response.diagnostics["code"], "verification_required")
        self.assertEqual(self.agent.snapshot("feedback").active_order, "18599253")

    def test_invalid_proof_format_and_vague_input_do_not_spend_retry(self):
        self.say("order 18599253")
        response = self.say("12345")
        self.assertEqual(response.diagnostics["code"], "invalid_verification_format")
        self.assertEqual(response.tool_events, ())
        response = self.say("Pincode 12345")
        self.assertEqual(response.diagnostics["code"], "invalid_verification_format")
        self.assertEqual(response.diagnostics["verification_attempts_used"], 0)
        response = self.say("what")
        self.assertEqual(response.diagnostics["waiting_for"], ["verification"])
        self.assertEqual(response.diagnostics["verification_attempts_used"], 0)
        self.assertIn("verification value", response.reply)
        self.assertEqual(self.say("4265").diagnostics["code"], "order_verified")

    def test_unclear_followup_does_not_repeat_verified_status(self):
        self.say("order 18599253")
        verified = self.say("4265")
        response = self.say("wjhat")
        self.assertEqual(response.decision, Decision.CLARIFY)
        self.assertNotEqual(response.reply, verified.reply)
        self.assertEqual(response.diagnostics["code"], "input_not_understood")
        self.assertTrue(self.agent.snapshot("feedback").verified)

    def test_guidance_is_localized(self):
        response = self.say("क्या")
        self.assertEqual(response.language, "hi")
        self.assertIn("समस्या", response.reply)
        self.say("मेरा ऑर्डर कहाँ है")
        response = self.say("१२३४")
        self.assertIn("4 अंक", response.reply)
        self.assertIn("8 अंकों", response.reply)

    def test_diagnostics_distinguish_backend_failure_from_wrong_proof(self):
        self.say("order 18599253")
        with patch.object(self.agent.operations.gateway.backend, "lookup_order", side_effect=OSError("private body")):
            response = self.say("4265")
        self.assertEqual(response.diagnostics["status"], "failed")
        self.assertEqual(response.diagnostics["code"], "lookup_failed")
        self.assertNotIn("private body", json.dumps(response.to_dict()))
        self.assertEqual(response.diagnostics["verification_attempts_used"], 0)
        response = self.say("0000")
        self.assertEqual(response.diagnostics["code"], "verification_failed")
        self.assertEqual(response.diagnostics["status"], "needs_input")

    def test_console_debug_status_reset_and_empty_input(self):
        output = self.console("", "wjhat", "new order", "1234", "/status", "/reset", "/status", "/bad", "/quit", debug=True)
        self.assertIn("AI generation is disabled", output)
        self.assertIn("Please type a support question", output)
        self.assertIn('"code": "invalid_order_id"', output)
        self.assertIn('"tools": []', output)
        self.assertIn('"llm_used": false', output)
        self.assertIn('"last_result": null', output)
        self.assertIn("Unknown command", output)
        self.assertIn("order verification has been cleared", output)
        self.assertNotIn("[debug]", self.console("what", "/quit"))

    def test_console_diagnostics_never_include_proofs_or_order_facts(self):
        output = self.console("order 18599253", "4265", "/status", "/quit", debug=True)
        self.assertNotIn("4265", output)
        debug = [json.loads(line[len("[debug] "):]) for line in output.splitlines() if line.startswith("[debug] ")]
        self.assertEqual(debug[-1]["code"], "order_verified")
        self.assertNotIn("verified_facts", str(debug))
        self.assertNotIn("customer_first_name", str(debug))

    def test_generation_failure_keeps_chat_open_and_does_not_echo_raw_error(self):
        self.agent.rag = Mock()
        self.agent.rag.respond.side_effect = LLMError("provider body with secret", code="connection_failed")
        output = self.console("How do coupons work?", "/status", "/quit", debug=True)
        self.assertIn("Policy generation failed", output)
        self.assertIn('"code": "connection_failed"', output)
        self.assertNotIn("provider body", output)
        self.assertIn('"stage": "policy_generation"', output)

    def test_cli_debug_flag_dispatches_without_network(self):
        from src.__main__ import main
        settings = Settings.load({"RUNTIME_DIR": str(Path(self.temp.name) / "cli")})
        output = StringIO()
        with patch("src.__main__.Settings.load", return_value=settings), patch("builtins.input", side_effect=["what", "/quit"]), redirect_stdout(output):
            self.assertEqual(main(["chat", "--debug"]), 0)
        self.assertIn("[debug]", output.getvalue())
        self.assertIn("input_not_understood", output.getvalue())


if __name__ == "__main__":
    unittest.main()
