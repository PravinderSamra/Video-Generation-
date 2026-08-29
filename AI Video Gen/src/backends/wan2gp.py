"""Wan2GP backend — the quality path, driven over Wan2GP's API.

Wan2GP ("AI video for the GPU-poor") wraps Wan 2.1/2.2, LTX, and Hunyuan with
quantisation and memory-offload profiles, which is what makes 14B-class models viable
on a consumer card. We use it for maximum quality; we use LTX directly for speed.

Setup:
    git clone https://github.com/deepbeepmeep/Wan2GP && cd Wan2GP
    ./install.sh          # or the Windows .bat, or Docker
    python wgp.py --api   # exposes the HTTP API
    echo "WAN2GP_URL=http://localhost:7860" >> "<project>/.env"

VERIFY BEFORE FIRST USE: Wan2GP ships roughly weekly, and its API route names have moved
between versions. `API_ROUTE` below matches the current documented shape; if a call 404s,
check the repo's API docs and override with WAN2GP_API_ROUTE. The offline batch path
(`python wgp.py --process <jobs.json>`) is the more stable alternative and is documented
in the README's troubleshooting section.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..config import EnrichedPrompt, GenerationRequest, Settings
from ._http import is_reachable, request_json
from .base import BackendError, RenderResult, VideoBackend

API_ROUTE = os.environ.get("WAN2GP_API_ROUTE", "/api/generate")
POLL_INTERVAL = 3.0
POLL_TIMEOUT = 3600


class Wan2GPBackend(VideoBackend):
    name = "wan2gp"

    def __init__(self, settings: Settings, model: str = "wan2.2-t2v-14b"):
        self.settings = settings
        self.model = model

    def preflight(self) -> list[str]:
        if not is_reachable(self.settings.wan2gp_url):
            return [
                f"Wan2GP not reachable at {self.settings.wan2gp_url} — "
                "start it with `python wgp.py --api`"
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

        base = self.settings.wan2gp_url.rstrip("/")
        started = time.monotonic()

        # We send the already-enriched prompt and explicitly disable Wan2GP's own
        # enhancer: enrichment happens once, in stage 1, where it is cached and
        # reviewable. Enriching twice drifts away from the reviewed text.
        response = request_json(
            f"{base}{API_ROUTE}",
            {
                "model": self.model,
                "prompt": prompt.enriched,
                "negative_prompt": prompt.negative,
                "resolution": f"{request.width}x{request.height}",
                "num_frames": request.num_frames,
                "fps": request.fps,
                "seed": request.seed,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance,
                "enhance_prompt": False,
            },
            timeout=self.settings.request_timeout,
        )

        produced = Path(self._resolve_output(base, response))
        if produced.is_file() and produced != output_path:
            produced.replace(output_path)
        elif not produced.is_file():
            raise BackendError(f"Wan2GP reported output at {produced}, but no file is there")

        return RenderResult(
            video_path=output_path,
            backend=self.name,
            model=self.model,
            elapsed_seconds=time.monotonic() - started,
            notes=["stage-1 enrichment used; Wan2GP's own enhancer disabled"],
        )

    def _resolve_output(self, base: str, response: dict) -> str:
        """Accept either a direct path or an async task id, and return a local path."""
        for key in ("video_path", "output_path", "path", "file"):
            if response.get(key):
                return response[key]

        task_id = response.get("task_id") or response.get("id")
        if not task_id:
            raise BackendError(f"Wan2GP response contained no output path or task id: {response}")

        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            status = request_json(f"{base}/api/status/{task_id}", timeout=30)
            state = str(status.get("status", "")).lower()
            if state in {"failed", "error"}:
                raise BackendError(f"Wan2GP task {task_id} failed: {status}")
            if state in {"done", "completed", "success"}:
                for key in ("video_path", "output_path", "path", "file"):
                    if status.get(key):
                        return status[key]
                raise BackendError(f"Wan2GP task {task_id} completed without a path: {status}")
            time.sleep(POLL_INTERVAL)
        raise BackendError(f"Wan2GP task {task_id} did not finish within {POLL_TIMEOUT}s")
