"""Configuration: paths, settings, and the request/result value objects.

Settings come from environment variables (optionally seeded from a .env file so a
checkout works without exporting anything). Nothing here requires a paid service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"


def _read_env_file(path: Path, override: bool) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def load_dotenv(path: Path | None = None) -> None:
    """Seed os.environ from .env, then let .env.local override it.

    .env follows the usual rule: real environment variables win. That breaks down when
    the environment itself holds a stale value you cannot change from where you are
    sitting — a hosted session pins its environment configuration at provisioning time,
    so a token you rotated afterwards never reaches the process, and every diagnostic
    points at the token rather than at the delivery. .env.local exists for exactly that:
    it wins over the environment. Keep it out of version control (it is gitignored).
    """
    _read_env_file(path or (ROOT / ".env"), override=False)
    _read_env_file(ROOT / ".env.local", override=True)


def _yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((PROMPTS_DIR / name).read_text(encoding="utf-8")) or {}


def load_style(name: str) -> str:
    """Resolve a style preset name to its description text."""
    styles = _yaml("style_presets.yaml")
    if name not in styles:
        available = ", ".join(sorted(styles))
        raise KeyError(f"unknown style {name!r}. Available: {available}")
    return " ".join(styles[name]["description"].split())


def load_dialect(name: str) -> str:
    """Resolve a target-model dialect hint. Falls back to the generic hint."""
    dialects = _yaml("model_dialects.yaml")
    entry = dialects.get(name) or dialects.get("generic", {})
    return " ".join(entry.get("hint", "").split())


def load_negative_prompt() -> str:
    raw = (PROMPTS_DIR / "negative_prompt.txt").read_text(encoding="utf-8")
    return " ".join(raw.split())


def load_system_prompt() -> str:
    """Read the enrichment system prompt, stripping the explanatory header."""
    raw = (PROMPTS_DIR / "cinematic_enrichment.md").read_text(encoding="utf-8")
    _, sep, body = raw.partition("\n---\n")
    return (body if sep else raw).strip()


@dataclass
class Settings:
    """Runtime endpoints and credentials. Every default points at a free resource."""

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    hf_token: str | None = None
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_text_model: str = "Qwen/Qwen3-8B"
    hf_video_model: str = "Wan-AI/Wan2.1-T2V-1.3B"  # smallest served model
    wan2gp_url: str = "http://localhost:7860"
    comfyui_url: str = "http://localhost:8188"
    ltx_repo: Path | None = None
    request_timeout: int = 180

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        ltx = os.environ.get("LTX_REPO")
        return cls(
            ollama_host=os.environ.get("OLLAMA_HOST", cls.ollama_host),
            ollama_model=os.environ.get("OLLAMA_MODEL", cls.ollama_model),
            hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
            hf_base_url=os.environ.get("HF_BASE_URL", cls.hf_base_url),
            hf_text_model=os.environ.get("HF_TEXT_MODEL", cls.hf_text_model),
            hf_video_model=os.environ.get("HF_VIDEO_MODEL", cls.hf_video_model),
            wan2gp_url=os.environ.get("WAN2GP_URL", cls.wan2gp_url),
            comfyui_url=os.environ.get("COMFYUI_URL", cls.comfyui_url),
            ltx_repo=Path(ltx).expanduser() if ltx else None,
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", cls.request_timeout)),
        )


@dataclass
class GenerationRequest:
    """Everything needed to render one shot, resolved before any GPU work starts."""

    prompt: str
    style: str = "cinematic"
    dialect: str = "generic"
    width: int = 768
    height: int = 512
    fps: int = 24
    duration: float = 5.0
    seed: int = 42
    steps: int = 30
    guidance: float = 3.0

    @property
    def num_frames(self) -> int:
        """Frame count, snapped to 8n+1 as LTX/Wan-family VAEs require."""
        raw = int(round(self.duration * self.fps))
        return max(9, ((raw - 1) // 8) * 8 + 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["num_frames"] = self.num_frames
        return data


@dataclass
class EnrichedPrompt:
    """The cacheable text artefact that sits between the two pipeline stages."""

    original: str
    enriched: str
    negative: str
    provider: str
    model: str
    style: str
    dialect: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
