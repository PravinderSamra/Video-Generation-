"""The pipeline: user prompt -> enriched prompt -> video file.

Two stages joined by one cacheable artefact. The enriched prompt is written to disk
*before* any GPU work begins, which is what lets you iterate on prompt engineering
(`--dry-run`) without paying for a render each time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import get_backend, slugify
from .config import OUTPUTS_DIR, EnrichedPrompt, GenerationRequest, Settings
from .enrich import get_enricher


@dataclass
class PipelineResult:
    request: GenerationRequest
    prompt: EnrichedPrompt
    video_path: Path | None
    sidecar_path: Path
    render: dict[str, Any] | None
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request": self.request.to_dict(),
            "enrichment": self.prompt.to_dict(),
            "render": self.render,
            "video_path": str(self.video_path) if self.video_path else None,
            "total_seconds": round(self.total_seconds, 2),
        }


def output_stem(request: GenerationRequest, outputs_dir: Path) -> Path:
    """A readable, collision-free stem: <slug>-<seed>[-N]."""
    base = outputs_dir / f"{slugify(request.prompt)}-{request.seed}"
    if not base.with_suffix(".json").exists():
        return base
    for index in range(2, 1000):
        candidate = outputs_dir / f"{base.name}-{index}"
        if not candidate.with_suffix(".json").exists():
            return candidate
    raise RuntimeError(f"cannot find a free output name for {base}")


def run(
    request: GenerationRequest,
    enricher_name: str = "ollama",
    backend_name: str = "ltx",
    dry_run: bool = False,
    outputs_dir: Path = OUTPUTS_DIR,
    settings: Settings | None = None,
    on_event=lambda message: None,
) -> PipelineResult:
    """Run both stages. With dry_run=True, stop after enrichment."""
    settings = settings or Settings.from_env()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(request, outputs_dir)
    started = time.monotonic()

    # ---- Stage 1: enrichment (text, CPU, seconds) --------------------------------
    on_event(f"enriching via {enricher_name} ...")
    prompt = get_enricher(enricher_name, settings).enrich(request)
    on_event("enriched.")

    render_info: dict[str, Any] | None = None
    video_path: Path | None = None

    # ---- Stage 2: generation (diffusion, GPU, minutes) ---------------------------
    if dry_run:
        on_event("dry run — stopping before generation.")
    else:
        backend = get_backend(backend_name, settings)
        for note in backend.warnings():
            on_event(f"  warning: {note}")
        on_event(f"rendering via {backend_name} ({backend.model}) ...")
        result = backend.render(request, prompt, stem.with_suffix(".mp4"))
        render_info = result.to_dict()
        video_path = result.video_path
        on_event(f"rendered in {result.elapsed_seconds:.1f}s -> {video_path}")

    pipeline_result = PipelineResult(
        request=request,
        prompt=prompt,
        video_path=video_path,
        sidecar_path=stem.with_suffix(".json"),
        render=render_info,
        total_seconds=time.monotonic() - started,
    )

    # Provenance sidecar: original prompt, enriched prompt, provider, model, seed,
    # and every resolved parameter. Enough to reproduce or compare this render.
    stem.with_suffix(".json").write_text(
        json.dumps(pipeline_result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return pipeline_result
