from __future__ import annotations

import os
from typing import Any

import httpx

from tinycoder.backends.base import Backend, BackendError


class OllamaBackend(Backend):
    """Ollama backend.

    Ollama's /api/chat accepts a `format` field that can be a full JSON schema
    (since v0.5). It constrains decoding via llama.cpp's grammar engine, so the
    response is guaranteed to parse — even for very small models.

    We override two defaults that bite small models hard:
      - num_ctx defaults to 2048 in Ollama, way too short once a long
        conversation builds up. We default to 16384.
      - num_predict defaults to 128 in some configs, which silently truncates
        long file contents mid-string. We default to 8192.

    Override via env: TINYCODER_NUM_CTX, TINYCODER_NUM_PREDICT.
    """

    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.num_ctx = int(os.environ.get("TINYCODER_NUM_CTX", "16384"))
        self.num_predict = int(os.environ.get("TINYCODER_NUM_PREDICT", "8192"))

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        try:
            r = self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"ollama request failed: {e}") from e

        data = r.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise BackendError(f"ollama returned empty content: {data!r}")
        return content
