from __future__ import annotations

import os
from typing import Any

import httpx

from tinycoder.backends.base import Backend, BackendError


class OpenAICompatBackend(Backend):
    """Generic OpenAI-compatible chat endpoint (LM Studio, vLLM, TGI, etc.).

    Uses response_format=json_schema when supported, falls back to json_object.
    Set strict_schema=False if your server only supports json_object.
    """

    name = "openai-compat"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        timeout: float = 300.0,
        strict_schema: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.strict_schema = strict_schema
        self._client = httpx.Client(timeout=timeout)
        self.max_tokens = int(os.environ.get("TINYCODER_NUM_PREDICT", "8192"))

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> str:
        if self.strict_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "agent_response", "schema": schema, "strict": True},
            }
        else:
            response_format = {"type": "json_object"}

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "response_format": response_format,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"openai-compat request failed: {e}") from e

        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"openai-compat returned malformed body: {data!r}") from e
        if not isinstance(content, str) or not content.strip():
            raise BackendError(f"openai-compat returned empty content: {data!r}")
        return content
