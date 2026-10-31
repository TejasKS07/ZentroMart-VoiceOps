"""Loopback-only web workspace using the same conversation controller as chat."""

from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import base64
import binascii
import shutil
from pathlib import Path
import secrets
from threading import RLock
import time
from urllib.parse import urlsplit
from tempfile import TemporaryDirectory

from .agent import AgentOutputError
from .llm import LLMError
from .schemas import CustomerInput

ASSETS = Path(__file__).resolve().parents[1] / "web"
COOKIE = "voiceops_session"
OWNER_COOKIE = "voiceops_browser"


class WebError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


@dataclass
class BrowserSession:
    touched: float = field(default_factory=time.monotonic)
    responses: dict = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)
    voice_turns: dict = field(default_factory=dict)


class WebApplication:
    def __init__(self, agent, *, voice=None, history_path=None, session_ttl=3600, max_sessions=128):
        self.agent = agent
        self.voice = voice
        self.session_ttl, self.max_sessions = session_ttl, max_sessions
        self.sessions = {}
        self.lock = RLock()
        from .history import HistoryStore
        self.history = HistoryStore(history_path)

    def status(self):
        return {"status": "ready", "mode": "live_policy" if self.agent.rag else "offline",
                "generation_enabled": self.agent.rag is not None, "voice_enabled": self.voice is not None and bool(shutil.which('ffmpeg')),
                "model": self.agent.rag.client.settings.llm_model if self.agent.rag else None,
                "languages": ["English", "हिंदी", "Hinglish"]}

    def new_session(self, previous=None, owner=None):
        with self.lock:
            now = time.monotonic()
            expired = [key for key, value in self.sessions.items() if now-value.touched > self.session_ttl or key == previous]
            for key in expired:
                session = self.sessions.pop(key)
                with session.lock:
                    self.agent.clear_session(key)
            if len(self.sessions) >= self.max_sessions:
                raise WebError(503, "workspace_busy", "The workspace is busy. Please try again shortly.")
            token = secrets.token_urlsafe(32)
            self.sessions[token] = BrowserSession()
            self.history.create(token, owner or secrets.token_urlsafe(32))
            return token

    def select_history(self, conversation, owner):
        record = self.history.get(conversation, owner)
        if not record:
            raise WebError(404, 'history_not_found', 'This conversation was not found.')
        with self.lock:
            existing = self.sessions.get(conversation)
            restored = existing is None or time.monotonic()-existing.touched > self.session_ttl
            if restored:
                self.agent.clear_session(conversation)
                self.sessions[conversation] = BrowserSession()
            else:
                existing.touched = time.monotonic()
        return {**record, 'verification_required_on_continue': restored, **self.status()}

    def chat(self, token, payload):
        if not isinstance(payload, dict) or set(payload) != {"message", "request_id"}:
            raise WebError(400, "invalid_request", "Send a message and a request ID.")
        message, request_id = payload["message"], payload["request_id"]
        if not isinstance(message, str) or not message.strip() or len(message) > 2000:
            raise WebError(400, "invalid_message", "Please enter a message between 1 and 2,000 characters.")
        if not isinstance(request_id, str) or not 16 <= len(request_id) <= 80 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in request_id):
            raise WebError(400, "invalid_request_id", "The message could not be identified. Please try again.")
        with self.lock:
            session = self.sessions.get(token)
            if session is None or time.monotonic()-session.touched > self.session_ttl:
                if session:
                    self.sessions.pop(token)
                    self.agent.clear_session(token)
                raise WebError(401, "session_expired", "This conversation expired. Start a new conversation to continue.")
            session.touched = time.monotonic()
            # Keep this session alive during in-flight generation/reset.
            with session.lock:
                if request_id in session.responses:
                    return session.responses[request_id]
                response = self.agent.respond(CustomerInput(message.strip(), token))
                data = self.agent.operations.gateway.sanitize(token, response.to_dict())
                # Raw model text is a development artifact, not browser content.
                data.pop("generation", None)
                result = {"response": data, "display_message": self.agent.operations.gateway.sanitize(token, message.strip())}
                session.responses[request_id] = result
                self.history.append(token, result, 'text:'+request_id)
                if len(session.responses) > 32:
                    del session.responses[next(iter(session.responses))]
                return result

    def voice_chat(self, token, payload):
        if self.voice is None:
            raise WebError(503, 'speech_not_configured', 'Gnani speech is not configured for this workspace.')
        if not isinstance(payload, dict) or set(payload) != {'audio_base64','extension','language','request_id'}:
            raise WebError(400, 'invalid_voice_request', 'Send audio, language, extension and request ID.')
        from .speech.audio import EXTENSIONS, MAX_BYTES
        from .speech.provider import LANGUAGES
        extension, language, request_id = payload['extension'], payload['language'], payload['request_id']
        if extension not in EXTENSIONS or language not in LANGUAGES or not isinstance(request_id,str) or not 16 <= len(request_id) <= 80 or not all(c.isascii() and (c.isalnum() or c in '-_') for c in request_id):
            raise WebError(400, 'invalid_voice_request', 'Choose a supported audio format and language.')
        try:
            if not isinstance(payload['audio_base64'],str):
                raise ValueError
            audio = base64.b64decode(payload['audio_base64'], validate=True)
            if not 0 < len(audio) <= MAX_BYTES:
                raise ValueError
        except (ValueError,binascii.Error):
            raise WebError(400, 'invalid_voice_audio', 'The audio upload is empty, invalid, or exceeds 20 MB.') from None
        with self.lock:
            session = self.sessions.get(token)
            if session is None or time.monotonic()-session.touched > self.session_ttl:
                raise WebError(401, 'session_expired', 'This conversation expired. Start a new conversation to continue.')
            session.touched = time.monotonic()
            with session.lock:
                turn = session.voice_turns.get(request_id)
                if turn and turn.response and turn.error and turn.error.get('stage') == 'tts':
                    turn = self.voice.synthesize(turn)
                elif turn is None:
                    with TemporaryDirectory(prefix='voiceops-upload-') as folder:
                        path = Path(folder) / ('recording' + extension)
                        path.write_bytes(audio)
                        turn = self.voice.process(path, token, language)
                if turn.response is None:
                    raise WebError(502, 'voice_turn_failed', 'The speech turn could not be completed. Please try again.')
                session.voice_turns[request_id] = turn
                if len(session.voice_turns)>4:
                    del session.voice_turns[next(iter(session.voice_turns))]
                data = self.agent.operations.gateway.sanitize(token, turn.to_dict())
                data['response'].pop('generation',None)
                result = {'response':data['response'], 'display_message':data['transcript'] or 'Audio message',
                        'voice':{k:data[k] for k in ('status','voice_complete','error','timings_ms','spoken_text')},
                        'reply_audio_base64':base64.b64encode(turn.audio).decode() if turn.audio else None}
                self.history.append(token, result, 'voice:'+request_id)
                return result


class SupportServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, application):
        self.application = application
        super().__init__(address, SupportHandler)
        port = self.server_address[1]
        self.allowed_hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
        self.allowed_origins = {"http://" + host for host in self.allowed_hosts}


class SupportHandler(BaseHTTPRequestHandler):
    server_version = "VoiceOps"

    def log_message(self, *_args):
        # Do not write proof-bearing request content or session cookies to logs.
        pass

    def _send(self, status, body, content_type="application/json; charset=utf-8", *, cookie=None, owner_cookie=None):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if owner_cookie:
            self.send_header('Set-Cookie', f'{OWNER_COOKIE}={owner_cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age=604800')
        if cookie:
            self.send_header("Set-Cookie", f"{COOKIE}={cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, error):
        self._send(error.status, {"error": {"code": error.code, "message": error.message}})

    def _check_host(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            raise WebError(403, "invalid_host", "Use the local workspace address.")

    def _token(self, name=COOKIE):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            return cookie[name].value if name in cookie else None
        except Exception:
            return None

    def do_GET(self):
        try:
            self._check_host()
            path = urlsplit(self.path).path
            if path == "/api/status":
                self._send(200, self.server.application.status())
                return
            if path == '/api/history':
                self._send(200, {'conversations':self.server.application.history.list(self._token(OWNER_COOKIE) or '')})
                return
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/recorder-worklet.js": ("recorder-worklet.js", "text/javascript; charset=utf-8"),
                      "/favicon.svg": ("favicon.svg", "image/svg+xml")}
            if path not in assets:
                raise WebError(404, "not_found", "This page was not found.")
            name, content_type = assets[path]
            self._send(200, (ASSETS / name).read_bytes(), content_type)
        except WebError as exc:
            self._error(exc)

    def do_POST(self):
        try:
            self._check_host()
            if self.headers.get("Origin") not in self.server.allowed_origins:
                raise WebError(403, "invalid_origin", "This request must come from the local workspace.")
            if self.headers.get_content_type() != "application/json":
                raise WebError(415, "invalid_content_type", "Use a JSON request.")
            if self.headers.get("Transfer-Encoding"):
                raise WebError(400, "invalid_body", "The request body is not supported.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise WebError(400, "invalid_body", "The request body is invalid.") from None
            if not 0 < length <= (28*1024*1024 if self.path == '/api/voice' else 16384):
                raise WebError(413, "message_too_large", "The message is too large.")
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                raise WebError(400, "invalid_json", "The request could not be read.") from None
            if self.path == "/api/session":
                if payload != {}:
                    raise WebError(400, "invalid_request", "Start a conversation with an empty request.")
                owner = self._token(OWNER_COOKIE) or secrets.token_urlsafe(32)
                token = self.server.application.new_session(self._token(), owner)
                self._send(201, {**self.server.application.status(),'conversation_id':token}, cookie=token, owner_cookie=owner)
            elif self.path == "/api/chat":
                self._send(200, self.server.application.chat(self._token(), payload))
            elif self.path == "/api/voice":
                self._send(200, self.server.application.voice_chat(self._token(), payload))
            elif self.path in {'/api/history/select','/api/history/delete'}:
                if not isinstance(payload,dict) or set(payload)!={'conversation_id'} or not isinstance(payload['conversation_id'],str):
                    raise WebError(400,'invalid_history_request','Choose a conversation.')
                owner = self._token(OWNER_COOKIE) or ''
                if self.path.endswith('/select'):
                    record = self.server.application.select_history(payload['conversation_id'],owner)
                    self._send(200,record,cookie=payload['conversation_id'])
                else:
                    if not self.server.application.history.delete(payload['conversation_id'],owner):
                        raise WebError(404,'history_not_found','This conversation was not found.')
                    self.server.application.agent.clear_session(payload['conversation_id'])
                    self.server.application.sessions.pop(payload['conversation_id'],None)
                    self._send(200,{'deleted':True})
            else:
                raise WebError(404, "not_found", "This endpoint was not found.")
        except WebError as exc:
            self._error(exc)
        except (LLMError, AgentOutputError):
            self._error(WebError(502, "generation_failed", "A valid policy answer could not be generated. Please try again."))
        except Exception:
            self._error(WebError(500, "request_failed", "The message could not be processed. Please try again."))


def serve(agent, *, host="127.0.0.1", port=8000, speech=None, history_path=None):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The local server must bind to 127.0.0.1 or localhost")
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    from .speech.engine import VoiceEngine
    with SupportServer((host, port), WebApplication(agent, voice=VoiceEngine(agent,speech) if speech else None,history_path=history_path)) as server:
        print(f"ZentroMart workspace: http://localhost:{server.server_address[1]}", flush=True)
        print("Press Ctrl+C to stop. " + ("Live policy generation enabled." if agent.rag else "Offline mode; AI generation disabled."), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
