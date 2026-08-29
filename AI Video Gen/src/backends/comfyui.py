"""ComfyUI backend — submit a workflow graph to a locally running ComfyUI server.

Use this when you want multi-step graphs (generate -> interpolate -> upscale) or a
persistent model cache across renders.

Setup:
    git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
    pip install -r requirements.txt && python main.py --listen
    # LTX nodes: clone Lightricks/ComfyUI-LTXVideo into custom_nodes/

LICENCE NOTE: ComfyUI is GPL-3.0. We speak to it only over its REST API, from a separate
process. Do not vendor or import its code into this project.

Workflow JSON lives in `workflows/` (API format — use ComfyUI's
"Save (API Format)" export, not the plain "Save"). Placeholders `%prompt%`,
`%negative%`, `%width%`, `%height%`, `%frames%`, `%seed%`, `%steps%` are substituted.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from ..config import ROOT, EnrichedPrompt, GenerationRequest, Settings
from ._http import is_reachable, request_json
from .base import BackendError, RenderResult, VideoBackend

WORKFLOWS_DIR = ROOT / "workflows"
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 1800


class ComfyUIBackend(VideoBackend):
    name = "comfyui"

    def __init__(self, settings: Settings, workflow: str = "ltx_t2v.json"):
        self.settings = settings
        self.workflow_path = WORKFLOWS_DIR / workflow
        self.model = f"comfyui:{Path(workflow).stem}"

    def preflight(self) -> list[str]:
        problems = []
        if not is_reachable(f"{self.settings.comfyui_url}/system_stats"):
            problems.append(
                f"ComfyUI not reachable at {self.settings.comfyui_url} — start it with "
                "`python main.py --listen`"
            )
        if not self.workflow_path.is_file():
            problems.append(
                f"workflow {self.workflow_path} not found — export one from ComfyUI "
                'using "Save (API Format)"'
            )
        return problems

    def _build_workflow(self, request: GenerationRequest, prompt: EnrichedPrompt) -> dict:
        raw = self.workflow_path.read_text(encoding="utf-8")
        substitutions = {
            "%prompt%": prompt.enriched,
            "%negative%": prompt.negative,
            "%width%": str(request.width),
            "%height%": str(request.height),
            "%frames%": str(request.num_frames),
            "%fps%": str(request.fps),
            "%seed%": str(request.seed),
            "%steps%": str(request.steps),
        }
        for key, value in substitutions.items():
            # json.dumps then strip quotes: escapes quotes/newlines inside the prompt.
            raw = raw.replace(key, json.dumps(value)[1:-1])
        return json.loads(raw)

    def render(
        self,
        request: GenerationRequest,
        prompt: EnrichedPrompt,
        output_path: Path,
    ) -> RenderResult:
        problems = self.preflight()
        if problems:
            raise BackendError("; ".join(problems))

        base = self.settings.comfyui_url.rstrip("/")
        client_id = str(uuid.uuid4())
        started = time.monotonic()

        queued = request_json(
            f"{base}/prompt",
            {"prompt": self._build_workflow(request, prompt), "client_id": client_id},
            timeout=self.settings.request_timeout,
        )
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise BackendError(f"ComfyUI did not return a prompt_id: {queued}")

        outputs = self._await_completion(base, prompt_id)
        saved = self._download_first_video(base, outputs, output_path)

        return RenderResult(
            video_path=saved,
            backend=self.name,
            model=self.model,
            elapsed_seconds=time.monotonic() - started,
            extra={"prompt_id": prompt_id},
        )

    def _await_completion(self, base: str, prompt_id: str) -> dict:
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            history = request_json(f"{base}/history/{prompt_id}", timeout=30)
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise BackendError(f"ComfyUI job failed: {status}")
                if entry.get("outputs"):
                    return entry["outputs"]
            time.sleep(POLL_INTERVAL)
        raise BackendError(f"ComfyUI job {prompt_id} did not finish within {POLL_TIMEOUT}s")

    @staticmethod
    def _download_first_video(base: str, outputs: dict, output_path: Path) -> Path:
        for node_output in outputs.values():
            for key in ("gifs", "videos", "images"):
                for item in node_output.get(key, []):
                    query = urllib.parse.urlencode(
                        {
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        }
                    )
                    with urllib.request.urlopen(f"{base}/view?{query}", timeout=300) as response:
                        output_path.write_bytes(response.read())
                    return output_path
        raise BackendError(f"no video found in ComfyUI outputs: {list(outputs)}")
