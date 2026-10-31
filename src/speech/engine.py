"""One speech turn, with honest partial failure and retryable synthesis."""

from dataclasses import dataclass, field, replace
from pathlib import Path
import time

from ..schemas import AgentResponse, CustomerInput, Decision
from .audio import SpeechError, normalize_audio
from .text import normalize_identifiers, spoken_reply


@dataclass
class VoiceTurn:
    status: str
    response: AgentResponse | None
    transcript: str = ""
    spoken_text: str = ""
    audio: bytes | None = field(default=None, repr=False)
    error: dict | None = None
    timings_ms: dict = field(default_factory=dict)
    audio_seconds: float | None = None
    understood: bool = True

    def to_dict(self):
        return {"status": self.status, "transcript": self.transcript, "spoken_text": self.spoken_text,
                "response": self.response.to_dict() if self.response else None, "error": self.error,
                "timings_ms": self.timings_ms, "audio_seconds": self.audio_seconds,
                "understood": self.understood,
                "voice_complete": self.status == "completed" and bool(self.audio) and self.understood}


class VoiceEngine:
    def __init__(self, agent, speech):
        self.agent, self.speech = agent, speech

    def _recovery(self, session, language, code):
        hi = language == "hi-IN"
        ambiguous = code == "ambiguous_identifier"
        reply = ("कृपया नंबर का हर अंक अलग-अलग दोहराएँ।" if ambiguous else "आवाज़ साफ़ नहीं आई। कृपया छोटा संदेश फिर रिकॉर्ड करें।") if hi else (
            "Please repeat the number one digit at a time." if ambiguous else "I could not hear a clear message. Please record a shorter message again.")
        if not self.agent.snapshot(session).introduced:
            reply=("मैं ZentroMart का AI सपोर्ट असिस्टेंट हूँ। " if hi else "I'm ZentroMart's AI support assistant. ")+reply
            self.agent.mark_introduced(session)
        return AgentResponse(Decision.CLARIFY, reply, language="hi" if hi else "hi-en" if language == "hi-en" else "en",
            diagnostics={"code": code, "status": "needs_input", "waiting_for": ["clear_audio"], "tools": [], "llm_used": False})

    def process(self, path: Path, session: str, language="en-IN", *, on_text=None):
        started = time.perf_counter()
        timings = {}
        transcript = ""
        response = None
        duration = None
        try:
            tick = time.perf_counter()
            audio, info = normalize_audio(path)
            duration = info.duration_seconds
            timings["audio"] = (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            transcript = self.speech.transcribe(audio, language)
            timings["stt"] = (time.perf_counter()-tick)*1000
            state = self.agent.snapshot(session)
            normalized, ambiguous = normalize_identifiers(transcript,
                expecting_identifier=state.missing_fields in {("order_id",), ("verification",)})
            if ambiguous:
                # Do not persist potentially spoken identity proof or guess digits.
                transcript = "[REDACTED: ambiguous identifier]"
                raise SpeechError("ambiguous_identifier", "stt", "Please repeat the identifier one digit at a time.")
            tick = time.perf_counter()
            response = self.agent.respond(CustomerInput(normalized, session))
            timings["agent"] = (time.perf_counter()-tick)*1000
            # Both raw spoken forms and normalized digits are secrets at boundaries.
            if state.missing_fields == ("verification",) or "last four" in normalized.lower() or "आख़िरी" in normalized:
                self.agent.operations.gateway.remember_proof(session, transcript)
            transcript = self.agent.operations.gateway.sanitize(session, normalized)
        except SpeechError as exc:
            response = self._recovery(session, language, exc.code)
            transcript = "" if exc.stage == "stt" and exc.code != "ambiguous_identifier" else transcript
            result = VoiceTurn("needs_repeat", response, transcript,
                error={"stage": exc.stage, "code": exc.code, "message": str(exc)}, timings_ms=timings, audio_seconds=duration, understood=False)
        except Exception:
            hi=language=='hi-IN'
            reply=('अभी प्रमाणित जवाब नहीं मिल पाया। कृपया फिर कोशिश करें या मानव सहयोगी से बात करने के लिए कहें।' if hi else
                'I could not produce a grounded answer just now. Please try again or ask for a human colleague.')
            if not self.agent.snapshot(session).introduced:
                reply=('मैं ZentroMart का AI सपोर्ट असिस्टेंट हूँ। ' if hi else "I'm ZentroMart's AI support assistant. ")+reply
                self.agent.mark_introduced(session)
            response=AgentResponse(Decision.CLARIFY,reply,language='hi' if hi else 'en',
                diagnostics={'code':'agent_failed','status':'failed','llm_used':False,'tools':[]})
            result=VoiceTurn('failed',response,'',error={'stage':'agent','code':'agent_failed','message':'The support turn could not be completed.'},
                timings_ms=timings,audio_seconds=duration,understood=False)
        else:
            result = VoiceTurn("text_ready", response, transcript, timings_ms=timings, audio_seconds=duration)
        result.spoken_text = spoken_reply(response.reply, response.language)
        if len(result.spoken_text) > 1200:
            # Do not read entire original Markdown tables as an improvised summary.
            result.spoken_text = "A detailed policy reference is available in the written reply. Please ask about one specific detail."
            if response.language == "hi":
                result.spoken_text = "लिखित जवाब में नीति का विस्तृत संदर्भ है। कृपया किसी एक बात के बारे में पूछें।"
        if on_text:
            on_text(result)
        result = self.synthesize(result)
        result.timings_ms["total"] = (time.perf_counter()-started)*1000
        return result

    def synthesize(self, result):
        tick = time.perf_counter()
        try:
            result.audio = self.speech.synthesize(result.spoken_text, result.response.language)
        except SpeechError as exc:
            result.status = "failed"
            result.error = {"stage": "tts", "code": exc.code, "message": str(exc)}
        except Exception:
            result.status = "failed"
            result.error = {"stage": "tts", "code": "tts_failed", "message": "Speech synthesis failed. The text reply is available."}
        else:
            if result.response.diagnostics.get('code')=='agent_failed':
                result.status='failed'
                result.error={'stage':'agent','code':'agent_failed','message':'The support turn could not be completed.'}
            elif not result.error or result.error.get("stage") == "tts":
                result.status, result.error = ("completed" if result.understood else "needs_repeat"), None
            # An audible clarification for unclear input is not a completed
            # understanding of the original clip; retain needs_repeat.
        result.timings_ms["tts"] = (time.perf_counter()-tick)*1000
        return result


def restored_turn(data):
    """Restore a sanitized text checkpoint for synthesis-only retry."""
    from ..schemas import Evidence, ToolEvent, GroundedClaim
    row = data['response']
    response = AgentResponse(Decision(row['decision']),row['reply'],
        evidence=tuple(Evidence(**e) for e in row.get('evidence',[])),
        retrieved_evidence_ids=tuple(row.get('retrieved_evidence_ids',[])),
        tool_events=tuple(ToolEvent(**e) for e in row.get('tool_events',[])),
        language=row.get('language','en'),timings_ms=row.get('timings_ms',{}),
        claims=tuple(GroundedClaim(c['text'],tuple(c['evidence_ids'])) for c in row.get('claims',[])),
        supported=row.get('supported'),diagnostics=row.get('diagnostics',{}))
    return VoiceTurn('text_ready',response,data.get('transcript',''),data.get('spoken_text',row['reply']),
        timings_ms=data.get('timings_ms',{}),audio_seconds=data.get('audio_seconds'),understood=data.get('understood',data.get('status')!='needs_repeat'))
