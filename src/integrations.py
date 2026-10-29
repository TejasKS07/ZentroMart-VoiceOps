"""Small connectivity checks, not the support agent or baseline implementation."""

import math
from pathlib import Path
import wave

from .config import Settings


class IntegrationError(RuntimeError):
    """Safe error message; provider response bodies must not be logged."""


def llm_smoke(settings: Settings) -> dict[str, object]:
    from .llm import ChatCompletionClient, LLMError
    from .config import ConfigurationError
    try:
        trace = ChatCompletionClient(settings).complete([
            {"role": "user", "content": "Reply with the word READY."}])
        if trace.finish_reason not in {"", "stop"}:
            raise IntegrationError("LLM generation was incomplete or blocked")
        return {"status": "passed", "reply_characters": len(trace.raw_output),
                "model": trace.model, "latency_ms": trace.latency_ms, "attempts": trace.attempts}
    except (LLMError, ConfigurationError) as exc:
        raise IntegrationError(str(exc)) from None


def gnani_smoke(settings: Settings, output_dir: Path,
                audio_path: Path | None = None) -> dict[str, object]:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import AudioConfig, GnaniTTSClient

    if not settings.gnani_api_key:
        raise IntegrationError("Set GNANI_API_KEY")
    if audio_path is not None and (not audio_path.is_file() or audio_path.stat().st_size == 0):
        raise IntegrationError("Input audio file is missing or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    reply_path = output_dir / "gnani_smoke.wav"
    try:
        tts = GnaniTTSClient(api_key=settings.gnani_api_key,
                             timeout=math.ceil(settings.timeout_seconds))
        audio = tts.synthesize(
            "Hello, this is a ZentroMart speech connectivity test.",
            model=settings.gnani_tts_model, voice=settings.gnani_tts_voice,
            language=settings.gnani_language,
            audio_config=AudioConfig(container="wav", encoding="linear_pcm", sample_rate=16000),
        )
        if not isinstance(audio, bytes) or not audio:
            raise IntegrationError("Gnani TTS returned no audio")
        reply_path.write_bytes(audio)
        with wave.open(str(reply_path), "rb") as handle:
            if not handle.getnframes():
                raise IntegrationError("Gnani TTS returned an empty WAV")
        stt = GnaniSTTClient(api_key=settings.gnani_api_key,
                             timeout=math.ceil(settings.timeout_seconds))
        result = stt.transcribe(str(audio_path or reply_path), language_code=settings.gnani_language)
        transcript = result.get("transcript", "")
        if not isinstance(transcript, str) or not transcript.strip():
            raise IntegrationError("Gnani STT returned no transcript")
        # Do not print or persist supplied recordings' possibly private transcript.
        return {"status": "passed", "audio_path": str(reply_path),
                "transcript_characters": len(transcript),
                "stt_input": "supplied_recording" if audio_path else "synthesized_loopback",
                "model": settings.gnani_tts_model, "voice": settings.gnani_tts_voice,
                "language": settings.gnani_language}
    except IntegrationError:
        raise
    except Exception:
        # SDK/network errors can include request text, keys or server payloads.
        raise IntegrationError("Gnani check failed; check credentials, model, voice, language and audio format") from None

