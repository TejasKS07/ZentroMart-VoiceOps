"""Real loopback HTTP checks for the support web boundary."""

import http.client
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch
from uuid import uuid4

from src.config import PROJECT_ROOT
from src.conversation import ConversationAgent
from src.tools.operations import SupportOperations
from src.web import SupportServer, WebApplication


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = SupportServer(("127.0.0.1", 0), None)
        cls.port = cls.server.server_address[1]
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agent = ConversationAgent(SupportOperations(PROJECT_ROOT / "Dataset", Path(self.temp.name)))
        self.server.application = WebApplication(self.agent)

    def request(self, path, payload=None, *, cookie=None, extra_headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        headers = {"Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        headers.update(extra_headers or {})
        connection.request("POST" if payload is not None else "GET", path,
                           body=json.dumps(payload) if payload is not None else None, headers=headers)
        response = connection.getresponse()
        status, response_headers, body = response.status, dict(response.getheaders()), response.read()
        response_headers['All-Set-Cookies'] = response.headers.get_all('Set-Cookie') or []
        connection.close()
        return status, response_headers, body

    def session(self, cookie=None):
        status, headers, _ = self.request("/api/session", {}, cookie=cookie)
        self.assertEqual(status, 201)
        return headers["Set-Cookie"].split(";", 1)[0]

    def chat(self, cookie, text, request_id=None):
        status, _, body = self.request("/api/chat", {"message": text, "request_id": request_id or uuid4().hex}, cookie=cookie)
        return status, json.loads(body)

    def test_website_assets_serve_without_private_files_or_external_dependencies(self):
        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Support, made simple.", body)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        for path in ("/app.js", "/styles.css", "/favicon.svg"):
            self.assertEqual(self.request(path)[0], 200)
        for path in ("/.env", "/Dataset/data/customers.csv", "/../src/config.py"):
            self.assertEqual(self.request(path)[0], 404)
        self.assertNotIn(b"gradient", self.request("/styles.css")[2])

    def test_status_and_cookie_expose_only_runtime_capabilities(self):
        status, _, body = self.request("/api/status")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["generation_enabled"])
        self.assertNotIn(b"api_key", body)
        self.assertNotIn(b"dataset_dir", body)
        _, headers, _ = self.request("/api/session", {})
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])

    def test_unknown_and_expired_sessions_do_not_accept_chat(self):
        self.assertEqual(self.chat(None, "Hi")[0], 401)
        cookie = self.session()
        token = cookie.split("=", 1)[1]
        self.server.application.sessions[token].touched -= 4000
        self.assertEqual(self.chat(cookie, "Hi")[0], 401)
        self.assertNotIn(token, self.server.application.sessions)

    def test_cross_origin_host_and_oversized_requests_are_rejected(self):
        self.assertEqual(self.request("/api/session", {}, extra_headers={"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("/api/status", extra_headers={"Host": "other.example"})[0], 403)
        self.assertEqual(self.chat(self.session(), "a" * 2001)[0], 400)
        self.assertEqual(self.request("/api/session", {"padding": "a" * 17000})[0], 413)
        self.assertEqual(self.request("/api/session", {}, extra_headers={"Content-Type": "text/plain"})[0], 415)

    def test_real_chat_validation_and_separate_browser_sessions(self):
        first, second = self.session(), self.session()
        self.chat(first, "Track my order")
        status, result = self.chat(first, "1234")
        self.assertEqual(status, 200)
        self.assertEqual(result["response"]["diagnostics"]["code"], "invalid_order_id")
        self.assertEqual(result["response"]["tool_events"], [])
        self.chat(first, "18599253")
        _, verified = self.chat(first, "4265")
        self.assertTrue(verified["response"]["diagnostics"]["verified"])
        _, unverified = self.chat(second, "Order 18599253")
        self.assertFalse(unverified["response"]["diagnostics"]["verified"])
        self.assertEqual(unverified["response"]["decision"], "clarify")
        self.assertNotIn("4265", json.dumps(verified))
        self.assertEqual(verified["display_message"], "[REDACTED]")
        self.assertNotIn("generation", verified["response"])

    def test_retry_does_not_consume_an_extra_identity_attempt(self):
        cookie = self.session()
        self.chat(cookie, "Order 18599253")
        request_id = uuid4().hex
        first = self.chat(cookie, "0000", request_id)
        again = self.chat(cookie, "0000", request_id)
        self.assertEqual(first, again)
        self.assertEqual(again[1]["response"]["diagnostics"]["verification_attempts_used"], 1)
        _, second = self.chat(cookie, "0000")
        self.assertEqual(second["response"]["decision"], "refuse")

    def test_new_conversation_clears_previous_identity(self):
        old = self.session()
        self.chat(old, "Order 18599253")
        self.chat(old, "4265")
        new = self.session(old)
        self.assertNotEqual(old, new)
        self.assertEqual(self.chat(old, "Where is my order?")[0], 401)
        _, response = self.chat(new, "Order 18599253")
        self.assertFalse(response["response"]["diagnostics"]["verified"])
        self.assertFalse(self.agent.snapshot(old.split("=", 1)[1]).verified)

    def test_server_errors_never_return_private_backend_details(self):
        cookie = self.session()
        with patch.object(self.agent, "respond", side_effect=RuntimeError("private secret body")):
            status, body = self.chat(cookie, "Hi")
        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["code"], "request_failed")
        self.assertNotIn("private secret", json.dumps(body))

    def test_voice_http_playable_output_and_synthesis_only_retry(self):
        from unittest.mock import Mock
        from test_speech import wav
        from src.speech.engine import VoiceEngine
        from src.speech.audio import SpeechError
        speech=Mock()
        speech.transcribe.return_value='Can I use two coupons?'
        speech.synthesize.side_effect=SpeechError('tts_failed','tts','Speech failed')
        self.server.application.voice=VoiceEngine(self.agent,speech)
        self.assertTrue(json.loads(self.request('/api/status')[2])['voice_enabled'])
        cookie=self.session()
        payload={'audio_base64':base64.b64encode(wav()).decode(),'extension':'.wav','language':'en-IN','request_id':uuid4().hex}
        status,_,body=self.request('/api/voice',payload,cookie=cookie)
        result=json.loads(body)
        self.assertEqual(status,200)
        self.assertEqual(result['voice']['status'],'failed')
        self.assertIsNone(result['reply_audio_base64'])
        speech.synthesize.side_effect=None;speech.synthesize.return_value=wav()
        result=json.loads(self.request('/api/voice',payload,cookie=cookie)[2])
        self.assertTrue(result['voice']['voice_complete'])
        self.assertTrue(base64.b64decode(result['reply_audio_base64']).startswith(b'RIFF'))
        self.assertEqual(speech.transcribe.call_count,1)

    def test_owner_scoped_history_and_restore_never_resurrect_identity(self):
        _,headers,body=self.request('/api/session',{})
        cookies='; '.join(value.split(';',1)[0] for value in headers['All-Set-Cookies'])
        conversation=json.loads(body)['conversation_id']
        self.chat(cookies,'Order 18599253');self.chat(cookies,'4265')
        result=json.loads(self.request('/api/history',cookie=cookies)[2])
        self.assertEqual(len(result['conversations']),1)
        self.assertEqual(result['conversations'][0]['title'],'Order support')
        self.request('/api/session',{},cookie=cookies)
        status,_,body=self.request('/api/history/select',{'conversation_id':conversation},cookie=cookies)
        self.assertEqual(status,200)
        self.assertTrue(json.loads(body)['verification_required_on_continue'])
        self.assertFalse(self.agent.snapshot(conversation).verified)
        self.assertNotIn('4265',body.decode())
        self.assertIn('[REDACTED]',body.decode())
        self.assertEqual(self.request('/api/history/select',{'conversation_id':conversation},cookie=self.session())[0],404)
        self.assertEqual(self.request('/api/history/delete',{'conversation_id':conversation},cookie=cookies)[0],200)
        self.assertEqual(self.request('/api/history/select',{'conversation_id':conversation},cookie=cookies)[0],404)


if __name__ == "__main__":
    unittest.main()
