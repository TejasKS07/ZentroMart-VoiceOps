"""Environment-based settings; dataset time is independent of wall-clock time."""

from dataclasses import dataclass, field
from datetime import date
import math
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATE = date(2026, 10, 20)


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    dataset_dir: Path
    runtime_dir: Path
    llm_endpoint: str = ""
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    gnani_api_key: str = field(default="", repr=False)
    gnani_tts_model: str = "timbre-v2.5"
    gnani_tts_voice: str = "Kaveri"
    gnani_language: str = "en-IN"
    timeout_seconds: float = 45
    llm_temperature: float | None = None
    llm_reasoning_effort: str = ""
    llm_calls_enabled: bool = False
    reference_date: date = field(default=REFERENCE_DATE, init=False)

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None,
             project_root: Path = PROJECT_ROOT, env_file: Path | None = None) -> "Settings":
        root = project_root.resolve()
        if env is None:
            dotenv_path = env_file or root / ".env"
            if not dotenv_path.is_absolute():
                dotenv_path = root / dotenv_path
            if env_file is not None and not dotenv_path.is_file():
                raise ConfigurationError("Requested environment file does not exist")
            if dotenv_path.is_file():
                try:
                    from dotenv import dotenv_values
                except ImportError as exc:
                    raise ConfigurationError("Install requirements.txt to read .env") from exc
                # Do not mutate the process environment; real env takes precedence.
                env = {**dotenv_values(dotenv_path), **os.environ}
            else:
                env = os.environ

        def value(name: str, default: str = "") -> str:
            return (env.get(name) or default).strip()

        def path(name: str, default: str) -> Path:
            candidate = Path(value(name, default)).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        endpoint = value("LLM_ENDPOINT")
        if endpoint:
            parsed = urlsplit(endpoint)
            local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or (parsed.scheme == "http" and not local)
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ConfigurationError("LLM_ENDPOINT needs HTTPS (or localhost HTTP), without credentials/query/fragment")
        try:
            timeout = float(value("API_TIMEOUT_SECONDS", "45"))
            if not math.isfinite(timeout) or not 0 < timeout <= 300:
                raise ValueError
        except ValueError as exc:
            raise ConfigurationError("API_TIMEOUT_SECONDS must be between 0 and 300") from exc
        temperature = None
        if value("LLM_TEMPERATURE"):
            try:
                temperature = float(value("LLM_TEMPERATURE"))
                if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                    raise ValueError
            except ValueError as exc:
                raise ConfigurationError("LLM_TEMPERATURE must be between 0 and 2, or blank") from exc
        effort = value("LLM_REASONING_EFFORT")
        if effort and effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ConfigurationError("Invalid LLM_REASONING_EFFORT")
        enabled = value("LLM_CALLS_ENABLED", "false").lower()
        if enabled not in {"true", "false"}:
            raise ConfigurationError("LLM_CALLS_ENABLED must be true or false")
        return cls(
            dataset_dir=path("DATASET_DIR", "Dataset"),
            runtime_dir=path("RUNTIME_DIR", "runtime"),
            llm_endpoint=endpoint, llm_model=value("LLM_MODEL"),
            llm_api_key=value("LLM_API_KEY"), gnani_api_key=value("GNANI_API_KEY"),
            gnani_tts_model=value("GNANI_TTS_MODEL", "timbre-v2.5"),
            gnani_tts_voice=value("GNANI_TTS_VOICE", "Kaveri"),
            gnani_language=value("GNANI_LANGUAGE", "en-IN"), timeout_seconds=timeout,
            llm_temperature=temperature, llm_reasoning_effort=effort,
            llm_calls_enabled=enabled == "true",
        )

    def generation_parameters(self) -> dict[str, object]:
        """Only explicitly configured parameters; provider defaults stay visible."""
        params: dict[str, object] = {}
        if self.llm_temperature is not None:
            params["temperature"] = self.llm_temperature
        if self.llm_reasoning_effort:
            params["reasoning_effort"] = self.llm_reasoning_effort
        return params

    def readiness(self) -> dict[str, object]:
        return {
            "dataset_dir": str(self.dataset_dir), "runtime_dir": str(self.runtime_dir),
            "reference_date": self.reference_date.isoformat(),
            "llm_configured": bool(self.llm_endpoint and self.llm_model),
            "llm_key_present": bool(self.llm_api_key),
            "gnani_key_present": bool(self.gnani_api_key),
            "llm_calls_enabled": self.llm_calls_enabled,
            "tts_model": self.gnani_tts_model, "tts_voice": self.gnani_tts_voice,
        }

