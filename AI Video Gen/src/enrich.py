"""Stage 1 — prompt enrichment.

Turns a short user idea into a dense, cinematic, technically detailed shot description.

This stage is deliberately decoupled from generation: it is a text problem, it runs on
CPU in seconds, and its output is a cached artefact you can inspect and diff. That is
what makes prompt engineering iterable without paying a render each time.

Providers (all free):
  ollama       local LLM on localhost:11434, no API key, works offline   [default]
  hf           Hugging Face Inference Providers, free monthly credits
  passthrough  no-op, for A/B comparison against an unenriched baseline
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .config import (
    EnrichedPrompt,
    GenerationRequest,
    Settings,
    load_dialect,
    load_negative_prompt,
    load_style,
    load_system_prompt,
)


class EnrichmentError(RuntimeError):
    """Raised when an enrichment provider is unreachable or returns nothing usable."""


def build_system_prompt(request: GenerationRequest) -> str:
    """Fill the template placeholders for this specific request."""
    return (
        load_system_prompt()
        .replace("{{style}}", load_style(request.style))
        .replace("{{dialect}}", load_dialect(request.dialect))
        .replace("{{duration}}", f"{request.duration:g}")
    )


def _chat_completion(
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    token: str | None = None,
) -> str:
    """Call any OpenAI-compatible /chat/completions endpoint.

    Ollama and the Hugging Face router both speak this shape, so one client covers
    the local and hosted paths with no provider-specific branching.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.8,
            "max_tokens": 500,
            "stream": False,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=payload, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise EnrichmentError(f"{base_url} returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EnrichmentError(f"cannot reach {base_url}: {exc}") from exc

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise EnrichmentError(f"unexpected response shape from {base_url}: {body}") from exc


def _clean(text: str) -> str:
    """Strip the wrappers instruct models add despite being told not to."""
    # Reasoning models may emit a <think> block before the answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    for fence in ("```markdown", "```text", "```"):
        if text.startswith(fence):
            text = text[len(fence):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    # Collapse to a single paragraph; the models sometimes add a stray trailing note.
    return " ".join(text.split())


class Enricher(ABC):
    """Stage-1 interface. Implement one method to add a provider."""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def enrich(self, request: GenerationRequest) -> EnrichedPrompt:
        ...

    def preflight(self) -> list[str]:
        """Cheap check for blocking problems. Empty means ready."""
        return []

    def warnings(self) -> list[str]:
        """Non-blocking notes."""
        return []

    def _wrap(self, request: GenerationRequest, text: str, notes: list[str]) -> EnrichedPrompt:
        return EnrichedPrompt(
            original=request.prompt,
            enriched=text,
            negative=load_negative_prompt(),
            provider=self.name,
            model=self.model,
            style=request.style,
            dialect=request.dialect,
            notes=notes,
        )


class PassthroughEnricher(Enricher):
    """No-op. The baseline you compare enriched output against."""

    name = "passthrough"
    model = "none"

    def enrich(self, request: GenerationRequest) -> EnrichedPrompt:
        return self._wrap(request, request.prompt, ["enrichment skipped (passthrough)"])


class OllamaEnricher(Enricher):
    """Local LLM via Ollama. Free, offline, no API key."""

    name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.ollama_model

    def preflight(self) -> list[str]:
        """Ollama is local, so check it is actually running before claiming ready."""
        try:
            with urllib.request.urlopen(
                f"{self.settings.ollama_host.rstrip('/')}/api/tags", timeout=3
            ) as response:
                installed = {
                    model["name"] for model in json.loads(response.read()).get("models", [])
                }
        except Exception:
            return [
                f"Ollama not reachable at {self.settings.ollama_host} — "
                "install from https://ollama.com and run `ollama serve`"
            ]
        if installed and self.model not in installed:
            return [f"model {self.model!r} not pulled — run `ollama pull {self.model}`"]
        return []

    def enrich(self, request: GenerationRequest) -> EnrichedPrompt:
        text = _chat_completion(
            base_url=f"{self.settings.ollama_host.rstrip('/')}/v1",
            model=self.model,
            system=build_system_prompt(request),
            user=request.prompt,
            timeout=self.settings.request_timeout,
        )
        return self._wrap(request, _clean(text), [])


class HFEnricher(Enricher):
    """Hosted LLM via Hugging Face Inference Providers (free monthly credits)."""

    name = "hf"

    def __init__(self, settings: Settings):
        if not settings.hf_token:
            raise EnrichmentError(
                "HF_TOKEN is not set. Create a free token at "
                "https://huggingface.co/settings/tokens and put it in .env"
            )
        self.settings = settings
        self.model = settings.hf_text_model

    def enrich(self, request: GenerationRequest) -> EnrichedPrompt:
        text = _chat_completion(
            base_url=self.settings.hf_base_url,
            model=self.model,
            system=build_system_prompt(request),
            user=request.prompt,
            timeout=self.settings.request_timeout,
            token=self.settings.hf_token,
        )
        return self._wrap(request, _clean(text), [])


ENRICHERS = {
    "ollama": OllamaEnricher,
    "hf": HFEnricher,
    "passthrough": PassthroughEnricher,
}


def get_enricher(name: str, settings: Settings) -> Enricher:
    if name not in ENRICHERS:
        raise EnrichmentError(f"unknown enricher {name!r}. Choose from: {', '.join(ENRICHERS)}")
    cls = ENRICHERS[name]
    return cls() if cls is PassthroughEnricher else cls(settings)
