from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendError(RuntimeError):
    """Raised when a backend call fails or returns un-parseable output."""


class Backend(ABC):
    """Abstract LLM backend that returns JSON conforming to a given schema."""

    name: str = "base"
    model: str

    @abstractmethod
    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> str:
        """Send chat messages and return the assistant's JSON string.

        Implementations MUST apply the JSON schema as a constrained-decoding
        grammar (or equivalent) so the returned string is always parseable.
        """
        raise NotImplementedError
