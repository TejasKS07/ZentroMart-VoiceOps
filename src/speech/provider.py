"""Gnani-only speech adapters with explicit settings and bounded safe retries."""

import math
import time
from uuid import uuid4

from .audio import SpeechError, inspect_wav

LANGUAGES = {"en-IN", "hi-IN", "hi-en", "indic-auto"}


class GnaniSpeech:
    def __init__(self, settings, *, stt=None, tts=None, sleep=time.sleep):
        if not settings.gnani_api_key:
            raise SpeechError("speech_not_configured", "configuration", "Set GNANI_API_KEY to enable speech.")
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        import logging
        logging.getLogger("gnani").setLevel(logging.CRITICAL)
        self.settings, self.sleep = settings, sleep
        self.stt = stt or GnaniSTTClient(api_key=settings.gnani_api_key, timeout=math.ceil(settings.timeout_seconds))
        self.tts = tts or GnaniTTSClient(api_key=settings.gnani_api_key, timeout=math.ceil(settings.timeout_seconds))

    def _call(self, stage, callback):
        import requests
        for attempt in range(2):
            try:
                return callback()
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = status in {408, 429, 500, 502, 503, 504} or isinstance(exc, (requests.Timeout, requests.ConnectionError))
                if transient and attempt == 0:
                    self.sleep(1)
                    continue
                code = f"{stage}_http_{status}" if isinstance(status, int) else f"{stage}_failed"
                raise SpeechError(code, stage, f"Gnani {stage.upper()} is unavailable. Please try again.") from None

    def transcribe(self, data: bytes, language="en-IN") -> str:
        if language not in LANGUAGES:
            raise SpeechError("unsupported_language", "stt", "Choose English, Hindi, or Hinglish.")
        # The pinned SDK supports explicit English/Hindi code-mix through en-IN.
        # Prisma's automatic mode is optional; model changes remain configuration.
        code = "en-IN" if language == "hi-en" else language
        kwargs = {"language_code": code, "model": self.settings.gnani_stt_model, "format": "transcribe", "request_id": uuid4().hex}
        if code == "indic-auto":
            kwargs["language_candidates"] = ["en-IN", "hi-IN"]
        result = self._call("stt", lambda: self.stt.transcribe_bytes(data, filename="audio.wav", **kwargs))
        if not isinstance(result, dict) or result.get("success") is False or result.get("truncated") is True:
            raise SpeechError("unclear_transcript", "stt", "The recording could not be transcribed completely. Please repeat a shorter message.")
        transcript = result.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip() or not any(c.isalnum() for c in transcript) or len(transcript) > 2000:
            raise SpeechError("empty_transcript", "stt", "I could not hear a clear message. Please record it again.")
        return transcript.strip()

    def synthesize(self, text: str, language="en") -> bytes:
        from gnani.tts import AudioConfig
        code = {"en": "en-IN", "hi": "hi-IN", "hi-en": "auto", "en-IN": "en-IN", "hi-IN": "hi-IN"}.get(language, "en-IN")
        data = self._call("tts", lambda: self.tts.synthesize(text, voice=self.settings.gnani_tts_voice,
            model=self.settings.gnani_tts_model, language=code,
            audio_config=AudioConfig(container="wav", encoding="linear_pcm", sample_rate=16000), request_id=uuid4().hex))
        if not isinstance(data, bytes):
            raise SpeechError("invalid_tts_audio", "tts", "Gnani returned no playable speech.")
        # Replies may last longer than caller clips; validate structure separately.
        import io, wave
        try:
            with wave.open(io.BytesIO(data), "rb") as stream:
                if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000 or not stream.getnframes():
                    raise ValueError
                samples = stream.readframes(stream.getnframes())
                if not samples or len(samples)%2 or len(samples)>16000*2*120:
                    raise ValueError
        except (ValueError, wave.Error, EOFError):
            raise SpeechError("invalid_tts_audio", "tts", "Gnani returned no playable speech.") from None
        # Gnani may return a streaming RIFF header with an unknown-length data
        # marker. Rebuild its header from the actual PCM payload before playback.
        output = io.BytesIO()
        with wave.open(output,'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
            stream.writeframes(samples)
        return output.getvalue()
