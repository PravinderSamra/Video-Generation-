"""Stage 2 — the video backend interface.

Every backend is a separate upstream project we talk to across a *stable seam*: a
documented CLI, or an HTTP API. We never import their internals, because they all move
fast and one of them (ComfyUI) is GPL-3.0.

Adding a backend means implementing `render()` and registering it. Nothing else changes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..config import EnrichedPrompt, GenerationRequest


class BackendError(RuntimeError):
    """Raised when a backend is unavailable or fails to produce a video."""


@dataclass
class RenderResult:
    """What a backend hands back. A path, never bytes."""

    video_path: Path
    backend: str
    model: str
    elapsed_seconds: float
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["video_path"] = str(self.video_path)
        return data


def slugify(text: str, max_length: int = 48) -> str:
    """A filesystem-safe stem derived from the prompt, for human-readable outputs."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_length].rstrip("-") or "shot")


class VideoBackend(ABC):
    """Stage-2 interface."""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def render(
        self,
        request: GenerationRequest,
        prompt: EnrichedPrompt,
        output_path: Path,
    ) -> RenderResult:
        """Render one shot to `output_path` and return a result describing it."""

    def preflight(self) -> list[str]:
        """Cheap check for *blocking* problems. Empty means the backend can run."""
        return []

    def warnings(self) -> list[str]:
        """Non-blocking notes — the backend will run, but degraded in some way."""
        return []
