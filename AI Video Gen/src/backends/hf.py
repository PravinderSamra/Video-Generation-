"""Hugging Face backend — hosted generation, no local GPU required.

The no-GPU escape hatch. Every free HF account gets monthly Inference Provider credits,
which is enough to prove the pipeline works before committing to a local model download.

Setup:
    free token at https://huggingface.co/settings/tokens
    echo "HF_TOKEN=hf_..." >> "<project>/.env"

VERIFY BEFORE FIRST USE: which video models are served, and by which provider, changes
over time. Check https://huggingface.co/models?pipeline_tag=text-to-video&inference_provider=all
for what is currently available on the free tier, then set HF_VIDEO_MODEL accordingly.
The router returns raw video bytes for text-to-video models.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import EnrichedPrompt, GenerationRequest, Settings
from ._http import request_bytes
from .base import BackendError, RenderResult, VideoBackend

ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"


class HFBackend(VideoBackend):
    name = "hf"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.hf_video_model

    def preflight(self) -> list[str]:
        if not self.settings.hf_token:
            return [
                "HF_TOKEN is not set — create a free token at "
                "https://huggingface.co/settings/tokens"
            ]
        return []

    def render(
        self,
        request: GenerationRequest,
        prompt: EnrichedPrompt,
        output_path: Path,
    ) -> RenderResult:
        problems = self.preflight()
        if problems:
            raise BackendError("; ".join(problems))

        started = time.monotonic()
        video = request_bytes(
            f"{ROUTER_BASE}/{self.model}",
            {
                "inputs": prompt.enriched,
                "parameters": {
                    "negative_prompt": prompt.negative,
                    "width": request.width,
                    "height": request.height,
                    "num_frames": request.num_frames,
                    "num_inference_steps": request.steps,
                    "guidance_scale": request.guidance,
                    "seed": request.seed,
                },
            },
            timeout=900,
            headers={"Authorization": f"Bearer {self.settings.hf_token}"},
        )

        if len(video) < 1024:
            raise BackendError(
                f"HF returned {len(video)} bytes — too small to be a video. "
                f"Is {self.model} served by an inference provider?"
            )
        output_path.write_bytes(video)

        return RenderResult(
            video_path=output_path,
            backend=self.name,
            model=self.model,
            elapsed_seconds=time.monotonic() - started,
            notes=["hosted inference — consumes free monthly HF credits"],
        )
