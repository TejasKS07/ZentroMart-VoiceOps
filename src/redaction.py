"""Redaction at persistence boundaries; secrets must be supplied by the caller."""

import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Iterable

MASK = "[REDACTED]"
PRIVATE_KEYS = {"phone", "phone_last4", "email", "address", "street_address",
                "delivery_address", "verification_value", "verification_proof",
                "llm_api_key", "gnani_api_key", "api_key", "authorization"}


def normalize_digits(text: str) -> str:
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in text)


def redact(value: Any, secrets: Iterable[str] = ()) -> Any:
    secrets = tuple(s for s in secrets if s)
    if isinstance(value, Mapping):
        return {str(k): MASK if str(k).lower() in PRIVATE_KEYS else redact(v, secrets)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        result = value
        for secret in secrets:
            if secret.isdecimal():
                # Handle ASCII/native-script digits and spoken digit grouping spaces.
                pattern = r"(?<!\d)" + r"[\s-]*".join(re.escape(c) for c in normalize_digits(secret)) + r"(?!\d)"
                normalized = normalize_digits(result)
                for match in reversed(list(re.finditer(pattern, normalized))):
                    result = result[:match.start()] + MASK + result[match.end():]
            else:
                result = result.replace(secret, MASK)
        return result
    return value

