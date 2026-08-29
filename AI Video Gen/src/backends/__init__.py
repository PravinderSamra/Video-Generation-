"""Backend registry. Add a backend here and the CLI picks it up automatically."""

from __future__ import annotations

from ..config import Settings
from .base import BackendError, RenderResult, VideoBackend, slugify
from .comfyui import ComfyUIBackend
from .hf import HFBackend
from .ltx import LTXBackend
from .stub import StubBackend
from .wan2gp import Wan2GPBackend

BACKENDS = {
    "ltx": LTXBackend,
    "wan2gp": Wan2GPBackend,
    "comfyui": ComfyUIBackend,
    "hf": HFBackend,
    "stub": StubBackend,
}

__all__ = [
    "BACKENDS",
    "BackendError",
    "RenderResult",
    "VideoBackend",
    "get_backend",
    "slugify",
]


def get_backend(name: str, settings: Settings) -> VideoBackend:
    if name not in BACKENDS:
        raise BackendError(f"unknown backend {name!r}. Choose from: {', '.join(BACKENDS)}")
    cls = BACKENDS[name]
    return cls() if cls is StubBackend else cls(settings)
