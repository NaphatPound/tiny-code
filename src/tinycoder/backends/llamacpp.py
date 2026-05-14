from __future__ import annotations

import os
from typing import Any

import httpx

from tinycoder.backends.base import Backend, BackendError


class LlamaCppBackend(Backend):
    """llama.cpp server backend (/v1/chat/completions, OpenAI-compatible route).

    llama.cpp supports `json_schema` in the response_format field, which is
    converted internally to a GBNF grammar. The `model` arg is mostly cosmetic
    since llama.cpp serves whichever GGUF was loaded at startup.

    We set max_tokens to 8192 (override with TINYCODER_NUM_PREDICT) so long
    file contents don't get truncated mid-string.
    """

    name = "llamacpp"

    def __init__(self, model: str, base_url: str = "http://localhost:8080", timeout: float = 300.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.max_tokens = int(os.environ.get("TINYCODER_NUM_PREDICT", "8192"))

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "agent_response", "schema": schema, "strict": True},
            },
        }
        try:
            r = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"llama.cpp request failed: {e}") from e

        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"llama.cpp returned malformed body: {data!r}") from e
        if not isinstance(content, str) or not content.strip():
            raise BackendError(f"llama.cpp returned empty content: {data!r}")
        return content
