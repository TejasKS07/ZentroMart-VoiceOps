"""Meaningful isolation, transport, artifact and grading regression checks."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.agent import AgentOutputError, BASELINE_PROMPT, BaselineAgent
from src.config import ConfigurationError, PROJECT_ROOT, Settings
from src.evaluation.cases import load_questions
from src.evaluation.judge import grade_run
from src.evaluation.runner import run_questions
from src.llm import ChatCompletionClient, LLMError
from src.schemas import AgentResponse, CustomerInput, Decision, GenerationTrace

DATASET = PROJECT_ROOT / "Dataset"


def trace(reply="I'm ZentroMart's AI support assistant. How can I help?", **overrides):
    return GenerationTrace(raw_output=json.dumps({"decision": "clarify", "reply": reply, "language": "en"}),
                           model="test-double-only", finish_reason="stop", **overrides)


class AgentTests(unittest.TestCase):
    def test_only_role_and_customer_text_are_sent_no_session_state_or_labels(self):
        client = Mock()
        client.complete.return_value = trace()
        agent = BaselineAgent(client)
        response = agent.respond(CustomerInput("my order is late", "test-session"))
        self.assertEqual(client.complete.call_args.args[0], [
            {"role": "system", "content": BASELINE_PROMPT}, {"role": "user", "content": "my order is late"}])
        self.assertEqual(response.evidence, ())
        self.assertEqual(response.tool_events, ())
        agent.respond(CustomerInput("new independent question", "test-session"))
        self.assertEqual(len(client.complete.call_args.args[0]), 2)

    def test_preserves_actual_reply_and_raw_output_without_repairing(self):
        client = Mock()
        client.complete.return_value = trace("An intentionally imperfect model reply")
        response = BaselineAgent(client).respond(CustomerInput("hello", "one"))
        self.assertEqual(response.reply, "An intentionally imperfect model reply")
        self.assertEqual(response.generation.raw_output, client.complete.return_value.raw_output)

    def test_rejects_invalid_or_truncated_output_without_regenerating(self):
        for output, reason in [("plain text", "stop"), ('{"decision":"answer"}', "stop"),
                               ('{"decision":"answer","reply":"x","language":"en","evidence":["RET-2"]}', "stop"),
                               (trace().raw_output, "length")]:
            with self.subTest(output=output):
                client = Mock()
                client.complete.return_value = GenerationTrace(output, "test-double", finish_reason=reason)
                with self.assertRaises(AgentOutputError) as error:
                    BaselineAgent(client).respond(CustomerInput("hello", "one"))
                self.assertEqual(error.exception.generation.raw_output, output)
                self.assertEqual(client.complete.call_count, 1)

    def test_json_fence_changes_only_container_not_customer_facing_reply(self):
        client = Mock()
        raw = "```json\n" + trace().raw_output + "\n```"
        client.complete.return_value = GenerationTrace(raw, "test-double")
        response = BaselineAgent(client).respond(CustomerInput("hi", "one"))
        self.assertEqual(response.generation.raw_output, raw)
        self.assertIn("AI support assistant", response.reply)


class TransportTests(unittest.TestCase):
    def settings(self):
        return Settings.load({"LLM_CALLS_ENABLED": "true", "LLM_ENDPOINT": "https://example.com/chat/completions", "LLM_MODEL": "test-model", "LLM_API_KEY": "secret"})

    @patch("src.llm.time.sleep")
    @patch("requests.post")
    def test_transient_http_retry_and_usage_capture(self, post, sleep):
        ok = Mock(status_code=200)
        ok.json.return_value = {"choices": [{"message": {"content": "READY"}, "finish_reason": "stop"}],
                                "model": "resolved-revision", "usage": {"prompt_tokens": 12, "completion_tokens": 2, "private": "secret"}}
        post.side_effect = [Mock(status_code=429), ok]
        result = ChatCompletionClient(self.settings()).complete([{"role": "user", "content": "test"}])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.model, "resolved-revision")
        self.assertEqual(result.usage, {"prompt_tokens": 12, "completion_tokens": 2})
        self.assertEqual(post.call_count, 2)
        self.assertFalse(post.call_args.kwargs["allow_redirects"])

    @patch("requests.post")
    def test_auth_error_is_not_retried_or_exposed(self, post):
        post.return_value = Mock(status_code=401, text="secret private server body")
        with self.assertRaises(LLMError) as error:
            ChatCompletionClient(self.settings()).complete([])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(error.exception.code, "http_401")
        self.assertNotIn("secret", str(error.exception))

    def test_generation_parameters_and_explicit_environment_file(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "settings.env"
            path.write_text('LLM_MODEL=chosen\nLLM_TEMPERATURE=0.2\nLLM_REASONING_EFFORT=low\n')
            with patch.dict("os.environ", {}, clear=True):
                settings = Settings.load(project_root=Path(folder), env_file=path)
            self.assertEqual(settings.llm_model, "chosen")
            self.assertEqual(settings.generation_parameters(), {"temperature": 0.2, "reasoning_effort": "low"})
        for env in ({"LLM_TEMPERATURE": "nan"}, {"LLM_REASONING_EFFORT": "unlimited"}):
            with self.assertRaises(ConfigurationError):
                Settings.load(env)


class RunTests(unittest.TestCase):
    def test_all_52_cases_use_real_input_boundary_and_save_distinct_raw_responses(self):
        cases = load_questions(DATASET)
        client = Mock()
        client.complete.side_effect = [trace(f"Unedited test-double response {i}") for i in range(52)]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "run"
            summary = run_questions(cases, BaselineAgent(client).respond, path)
            rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
            self.assertEqual(summary["completed"], 52)
            self.assertEqual(summary["fact_correctness"], "not_scored")
            self.assertEqual(rows[-1]["response"]["reply"], "Unedited test-double response 51")
        for call, case in zip(client.complete.call_args_list, cases):
            messages = call.args[0]
            self.assertEqual(messages[1]["content"], case.customer.text)
            self.assertEqual(messages[0]["content"], BASELINE_PROMPT)
            self.assertEqual(len(messages), 2)

    def test_interruption_preserves_partial_results_and_unprocessed_count(self):
        cases = load_questions(DATASET)[:2]
        calls = 0

        def respond(customer):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return AgentResponse(Decision.ANSWER, "test double")

        with TemporaryDirectory() as folder:
            path = Path(folder) / "run"
            with self.assertRaises(KeyboardInterrupt):
                run_questions(cases, respond, path)
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual(summary["run_status"], "interrupted")
            self.assertEqual(summary["processed"], 1)
            self.assertEqual(summary["cases"], 2)
            self.assertEqual(len((path / "results.jsonl").read_text().splitlines()), 1)

    def test_bad_model_output_is_saved_for_diagnosis(self):
        client = Mock()
        client.complete.return_value = GenerationTrace("not-json", "test-double")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "run"
            run_questions(load_questions(DATASET)[:1], BaselineAgent(client).respond, path)
            row = json.loads((path / "results.jsonl").read_text())
            self.assertEqual(row["error_code"], "invalid_agent_output")
            self.assertEqual(row["generation"]["raw_output"], "not-json")

    def test_repeated_service_outage_halts_without_scoring_unprocessed_questions(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "run"
            def unavailable(customer):
                raise LLMError("Service unavailable", code="http_503")
            summary = run_questions(load_questions(DATASET), unavailable, path)
            self.assertEqual(summary["run_status"], "halted_service_failure")
            self.assertEqual(summary["processed"], 3)
            self.assertEqual(summary["cases"], 52)
            self.assertEqual(summary["fact_correctness"], "not_scored")

    def test_service_failure_makes_semantic_score_unresolved(self):
        cases = load_questions(DATASET)[:1]
        client = Mock()
        client.settings = Settings.load({"LLM_MODEL": "test-double"})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "run"
            def unavailable(customer):
                raise LLMError("Service unavailable", code="http_503")
            run_questions(cases, unavailable, path)
            summary = grade_run(path, cases, client)
            self.assertIsNone(summary["accuracy"])
            self.assertEqual(summary["unresolved"], 1)
            self.assertEqual(summary["status"], "incomplete")
            client.complete.assert_not_called()

    def test_semantic_grade_preserves_answers_and_uses_boolean_checks(self):
        cases = load_questions(DATASET)[:1]
        client = Mock()
        client.settings = Settings.load({"LLM_ENDPOINT": "https://example.com/chat/completions", "LLM_MODEL": "test-double"})
        client.complete.return_value = GenerationTrace(json.dumps({
            "facts_correct": False, "forbidden_claims_absent": True,
            "language_correct": True, "ai_disclosed": True, "rationale": "Required fact is missing."}), "test-double")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "run"
            run_questions(cases, lambda c: AgentResponse(Decision.ANSWER, "Original actual response"), path)
            before = (path / "results.jsonl").read_bytes()
            summary = grade_run(path, cases, client)
            self.assertEqual(summary["accuracy"], 0)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual((path / "results.jsonl").read_bytes(), before)
            with self.assertRaises(FileExistsError):
                grade_run(path, cases, client)


if __name__ == "__main__":
    unittest.main()
