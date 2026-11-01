"""Bounded audio decoding; ffmpeg is a local codec, never a speech provider."""

from dataclasses import dataclass
import io
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import wave

import numpy as np

EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 30


class SpeechError(RuntimeError):
    def __init__(self, code, stage, message):
        super().__init__(message)
        self.code, self.stage = code, stage


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: float
    rms: float
    sample_rate: int


def inspect_wav(data: bytes, *, reject_silence=False) -> AudioInfo:
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
                raise ValueError
            duration = stream.getnframes() / stream.getframerate()
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64)
            if len(samples) != stream.getnframes() or not 0.1 <= duration <= MAX_SECONDS:
                raise ValueError
    except (ValueError, wave.Error, EOFError):
        raise SpeechError("invalid_audio", "audio", "Use a readable audio clip between 0.1 and 30 seconds.") from None
    rms = float(np.sqrt(np.mean(samples*samples)))
    if reject_silence and rms < 60:
        raise SpeechError("silent_audio", "audio", "The recording is silent or too quiet. Please record again closer to the microphone.")
    return AudioInfo(duration, rms, 16000)


def normalize_audio(path: Path) -> tuple[bytes, AudioInfo]:
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS or not 0 < path.stat().st_size <= MAX_BYTES:
        raise SpeechError("unsupported_audio", "audio", "Use a supported audio file of at most 20 MB.")
    if not shutil.which("ffmpeg"):
        raise SpeechError("decoder_missing", "audio", "Install ffmpeg to decode audio files.")
    with TemporaryDirectory(prefix="voiceops-audio-") as folder:
        output = Path(folder) / "normalized.wav"
        try:
            process = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path.resolve()),
                "-vn", "-t", "31", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(output)],
                capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            raise SpeechError("decode_failed", "audio", "The recording could not be decoded. Please try another file.") from None
        if process.returncode or not output.is_file():
            raise SpeechError("decode_failed", "audio", "The recording could not be decoded. Please try another file.")
        data = output.read_bytes()
    return data, inspect_wav(data, reject_silence=True)
