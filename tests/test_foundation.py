"""Offline regression checks; doubles never count as live API validation."""

from contextlib import redirect_stdout
from datetime import date
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import wave

from src.__main__ import main
from src.config import ConfigurationError, PROJECT_ROOT, Settings
from src.dataset import DatasetError, PublicKnowledge, read_csv, read_document
from src.evaluation.cases import load_conversations, load_questions
from src.evaluation.runner import run_questions
from src.integrations import IntegrationError, gnani_smoke, llm_smoke
from src.redaction import redact
from src.schemas import AgentResponse, Decision, Evidence, ToolEvent

DATASET = PROJECT_ROOT / "Dataset"


class ConfigurationTests(unittest.TestCase):
    def test_fixed_reference_date_and_root_relative_paths(self):
        settings = Settings.load({"DATASET_DIR": "external/data"}, Path("/tmp/project"))
        self.assertEqual(settings.reference_date, date(2026, 10, 20))
        self.assertEqual(settings.dataset_dir, Path("/tmp/project/external/data"))

    def test_credentials_do_not_appear_in_repr_or_readiness(self):
        settings = Settings.load({"LLM_API_KEY": "private-llm", "GNANI_API_KEY": "private-speech"})
        snapshot = repr(settings) + json.dumps(settings.readiness())
        self.assertNotIn("private-llm", snapshot)
        self.assertNotIn("private-speech", snapshot)

    def test_env_overrides_dotenv_without_mutating_process(self):
        with TemporaryDirectory() as folder, patch.dict("os.environ", {"LLM_MODEL": "from-env"}, clear=True):
            root = Path(folder)
            (root / ".env").write_text("LLM_MODEL=from-file\nGNANI_API_KEY=file-key\n")
            settings = Settings.load(project_root=root)
            self.assertEqual(settings.llm_model, "from-env")
            self.assertEqual(settings.gnani_api_key, "file-key")
            import os
            self.assertNotIn("GNANI_API_KEY", os.environ)

    def test_invalid_timeout_and_endpoint_are_rejected(self):
        for timeout in ("0", "-1", "nan", "inf", "301", "oops"):
            with self.subTest(timeout=timeout), self.assertRaises(ConfigurationError):
                Settings.load({"API_TIMEOUT_SECONDS": timeout})
        for endpoint in ("http://remote.example/v1", "https://user:pass@example.com", "https://example.com?key=secret"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ConfigurationError):
                Settings.load({"LLM_ENDPOINT": endpoint})
        self.assertEqual(Settings.load({"LLM_ENDPOINT": "http://localhost:8000/v1/chat/completions"}).timeout_seconds, 45)


class DatasetTests(unittest.TestCase):
    def test_preserves_leading_zero_identifiers_and_missing_cells(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            path.write_text("order_id,pincode,rrn,missing\n00123456,012345,000001,\n")
            row = read_csv(path, ("order_id",))[0]
            self.assertEqual(row, {"order_id": "00123456", "pincode": "012345", "rrn": "000001", "missing": ""})

    def test_rejects_malformed_rows_and_columns(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            for content in ("id,id\n1,2\n", "id,name\n1\n", "id\n1,2\n"):
                path.write_text(content)
                with self.assertRaises(DatasetError):
                    read_csv(path, ("id",))

    def test_public_knowledge_has_no_private_tables_or_practice_answers(self):
        knowledge = PublicKnowledge.load(DATASET)
        self.assertEqual(len(knowledge.products), 70)
        self.assertEqual(len(knowledge.documents), 21)
        self.assertEqual(set(vars(knowledge)), {"products", "pincodes", "holidays", "documents"})

    def test_section_metadata_tables_and_historical_policy_survive_loading(self):
        doc = read_document(DATASET / "kb/pol_returns_v4.md")
        sections = {section.evidence_id: section for section in doc.sections}
        self.assertEqual(sections["RET-2"].metadata["status"], "current")
        self.assertIn("| Electronics and accessories | 7 days |", sections["RET-2"].text)
        self.assertIn("before 1 August 2026", sections["RET-3"].text)
        historical = read_document(DATASET / "kb/pol_returns_v3_superseded.md")
        self.assertEqual(historical.metadata["status"], "superseded")


class ContractTests(unittest.TestCase):
    def test_rejects_bad_decision_empty_reply_and_invalid_timing(self):
        for kwargs in ({"decision": "invent", "reply": "x"},
                       {"decision": "answer", "reply": " "},
                       {"decision": "answer", "reply": "x", "timings_ms": {"llm": float("nan")}}):
            with self.assertRaises(ValueError):
                AgentResponse(**kwargs)

    def test_serialized_response_contains_evidence_and_redacts_verification(self):
        evidence = Evidence("RET-2", "kb/pol_returns_v4.md", "Policy section text")
        response = AgentResponse(Decision.CLARIFY, "Proof received: ४ ५ ३ ३.", evidence=(evidence,),
                                 tool_events=(ToolEvent("lookup_order", "succeeded", {
                                     "order_id": "00123456", "verified": True,
                                     "verification_value": "4533", "phone": "+91 secret",
                                     "items": [{"title": "private"}], "summary": "private",
                                 }),))
        result = response.to_dict(secrets=("4533",))
        self.assertEqual(result["evidence_ids"], ["RET-2"])
        self.assertEqual(result["reply"], "Proof received: [REDACTED].")
        self.assertEqual(result["tool_events"][0]["details"], {"order_id": "00123456", "verified": True})
        self.assertNotIn("private", json.dumps(result))

    def test_nested_key_and_secret_redaction(self):
        self.assertEqual(redact({"payload": {"verification_value": "1234"}, "text": "key abc-token"},
                                ("abc-token",)),
                         {"payload": {"verification_value": "[REDACTED]"}, "text": "key [REDACTED]"})


class EvaluationTests(unittest.TestCase):
    def test_practice_counts_ordering_and_asr(self):
        questions = load_questions(DATASET)
        asr = load_questions(DATASET, asr=True)
        self.assertEqual(len(questions), 52)
        self.assertNotEqual(questions[0].customer.text, asr[0].customer.text)
        groups = load_conversations(DATASET)
        self.assertEqual(len(groups), 23)
        self.assertEqual(sum(len(cases) for cases in groups.values()), 96)
        for cases in groups.values():
            self.assertEqual([case.turn for case in cases], list(range(1, len(cases)+1)))
            self.assertNotIn("order_id", vars(cases[0].customer))

    def test_runner_passes_no_labels_and_continues_after_failure(self):
        cases = load_questions(DATASET)[:2]
        seen = []

        def test_double(customer):
            seen.append(customer)
            self.assertEqual(set(vars(customer)), {"text", "session_id"})
            if len(seen) == 1:
                raise RuntimeError("Do not save private exception contents")
            return AgentResponse(Decision.ANSWER, "Test double response")

        with TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            summary = run_questions(cases, test_double, output)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["fact_correctness"], "not_scored")
            self.assertNotIn("private exception", (output / "results.jsonl").read_text())
            with self.assertRaises(FileExistsError):
                run_questions(cases, test_double, output)

    def test_missing_credentials_are_pending_not_passed(self):
        stream = StringIO()
        with patch("src.__main__.Settings.load", return_value=Settings.load({})), redirect_stdout(stream):
            self.assertEqual(main(["smoke"]), 2)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["checks"]["llm"]["status"], "pending")
        self.assertEqual(result["checks"]["gnani"]["status"], "pending")


class IntegrationTests(unittest.TestCase):
    def settings(self, **overrides):
        return Settings.load({"LLM_CALLS_ENABLED": "true", "LLM_ENDPOINT": "https://example.com/v1/chat/completions",
                              "LLM_MODEL": "configured-model", "LLM_API_KEY": "secret",
                              "GNANI_API_KEY": "speech-secret", **overrides})

    @patch("requests.post")
    def test_llm_contract_and_credential_forwarding(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": [{"message": {"content": "READY"}}]}
        result = llm_smoke(self.settings())
        self.assertEqual(result["status"], "passed")
        kwargs = post.call_args.kwargs
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kwargs["json"]["model"], "configured-model")

    @patch("requests.post")
    def test_provider_errors_do_not_echo_raw_bodies(self, post):
        post.return_value = Mock(status_code=401, text="secret and private request")
        with self.assertRaisesRegex(IntegrationError, "HTTP 401") as error:
            llm_smoke(self.settings())
        self.assertNotIn("secret", str(error.exception))

    @patch("gnani.stt.GnaniSTTClient")
    @patch("gnani.tts.GnaniTTSClient")
    def test_gnani_explicit_model_wav_and_loopback(self, tts, stt):
        buffer = BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\0\0" * 1600)
        tts.return_value.synthesize.return_value = buffer.getvalue()
        stt.return_value.transcribe.return_value = {"transcript": "test"}
        with TemporaryDirectory() as folder:
            result = gnani_smoke(self.settings(), Path(folder))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["stt_input"], "synthesized_loopback")
            self.assertTrue(Path(result["audio_path"]).is_file())
            self.assertEqual(tts.return_value.synthesize.call_args.kwargs["model"], "timbre-v2.5")
            self.assertNotIn("transcript", result)


if __name__ == "__main__":
    unittest.main()

