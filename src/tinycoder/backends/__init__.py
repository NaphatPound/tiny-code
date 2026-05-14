from tinycoder.backends.base import Backend, BackendError
from tinycoder.backends.ollama import OllamaBackend
from tinycoder.backends.llamacpp import LlamaCppBackend
from tinycoder.backends.openai_compat import OpenAICompatBackend

__all__ = [
    "Backend",
    "BackendError",
    "OllamaBackend",
    "LlamaCppBackend",
    "OpenAICompatBackend",
    "build_backend",
]


def build_backend(name: str, model: str, base_url: str | None = None, api_key: str | None = None) -> Backend:
    name = name.lower()
    if name == "ollama":
        return OllamaBackend(model=model, base_url=base_url or "http://localhost:11434")
    if name in ("llamacpp", "llama.cpp", "llama-cpp"):
        return LlamaCppBackend(model=model, base_url=base_url or "http://localhost:8080")
    if name in ("openai", "openai-compat", "lmstudio", "vllm"):
        return OpenAICompatBackend(
            model=model,
            base_url=base_url or "http://localhost:1234/v1",
            api_key=api_key or "not-needed",
        )
    raise ValueError(f"unknown backend: {name!r}")
