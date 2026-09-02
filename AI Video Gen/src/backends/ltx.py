"""LTX-Video backend — local generation via the upstream `inference.py` CLI.

Default backend: best quality-per-VRAM-per-second of the options evaluated, Apache-2.0,
and the only one with a clean documented single-command CLI.

Setup:
    git clone https://github.com/Lightricks/LTX-Video && cd LTX-Video
    python -m pip install -e .[inference]
    echo "LTX_REPO=$(pwd)" >> "<project>/.env"

We shell out rather than importing, so upstream refactors cannot break us.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from ..config import EnrichedPrompt, GenerationRequest, Settings
from .base import BackendError, RenderResult, VideoBackend

# Pipeline configs shipped by the upstream repo, cheapest first.
PIPELINE_CONFIGS = {
    "2b-distilled": "configs/ltxv-2b-0.9.8-distilled.yaml",
    "13b-distilled": "configs/ltxv-13b-0.9.8-distilled.yaml",
    "13b-dev": "configs/ltxv-13b-0.9.8-dev.yaml",
}


class LTXBackend(VideoBackend):
    name = "ltx"

    def __init__(self, settings: Settings, variant: str | None = None):
        # The variant decides whether the weights fit the card. A 16 GB GPU (Kaggle's
        # free T4, and Colab's) holds 2b-distilled but not 13b, so this has to be
        # selectable without editing code — set LTX_VARIANT, as .env.local does.
        variant = variant or settings.ltx_variant
        if variant not in PIPELINE_CONFIGS:
            raise BackendError(
                f"unknown LTX variant {variant!r}. Choose from: {', '.join(PIPELINE_CONFIGS)}"
            )
        self.settings = settings
        self.variant = variant
        self.model = f"ltxv-{variant}"

    def preflight(self) -> list[str]:
        repo = self.settings.ltx_repo
        if repo is None:
            return ["LTX_REPO is not set — clone Lightricks/LTX-Video and point .env at it"]
        if not (repo / "inference.py").is_file():
            return [f"{repo}/inference.py not found — is LTX_REPO pointing at the repo root?"]
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

        repo = self.settings.ltx_repo
        assert repo is not None  # guaranteed by preflight

        with tempfile.TemporaryDirectory() as tmp:
            command = [
                "python", "inference.py",
                "--prompt", prompt.enriched,
                "--negative_prompt", prompt.negative,
                "--height", str(request.height),
                "--width", str(request.width),
                "--num_frames", str(request.num_frames),
                "--frame_rate", str(request.fps),
                "--seed", str(request.seed),
                "--pipeline_config", str(self._pipeline_config(repo, Path(tmp))),
                "--output_path", str(output_path.parent),
                # Upstream self-disables this above 30 GB of VRAM, so it is safe to pass
                # always: it is what makes a 16 GB card (Kaggle, Colab) fit at all.
                "--offload_to_cpu",
            ]

            started = time.monotonic()
            # st_mtime is wall-clock, so _newest_video needs a wall-clock marker, not the
            # monotonic one used for elapsed. Back it off a second for filesystem timestamp
            # granularity, which is coarser than a float second on some filesystems.
            started_wall = time.time() - 1
            result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise BackendError(
                f"LTX inference.py exited {result.returncode}:\n{result.stderr.strip()[-1500:]}"
            )

        produced = self._newest_video(output_path.parent, started_wall)
        if produced is None:
            raise BackendError(
                "LTX reported success but no video appeared in "
                f"{output_path.parent}. stdout tail:\n{result.stdout.strip()[-800:]}"
            )
        if produced != output_path:
            produced.replace(output_path)

        return RenderResult(
            video_path=output_path,
            backend=self.name,
            model=self.model,
            elapsed_seconds=elapsed,
            notes=[f"pipeline_config={PIPELINE_CONFIGS[self.variant]}"],
        )

    def _pipeline_config(self, repo: Path, tmp: Path) -> Path:
        """Upstream's pipeline config, with its prompt enhancer switched off.

        LTX re-enriches any prompt shorter than `prompt_enhancement_words_threshold`
        words by loading Florence-2-large and Llama-3.2-3B onto the *same* card as the
        transformer. That is wrong for us twice over. It is ~8 GB on top of a budget
        that already only just fits 16 GB — the OOM a Kaggle T4 dies on. And this
        pipeline has already enriched: stage one produced `prompt.enriched`, and the
        sidecar records it as what was rendered. Letting LTX rewrite the prompt after
        that makes the sidecar describe something other than the clip beside it, which
        is the one claim the sidecar exists to make.

        The threshold is only readable from the config file — there is no CLI override —
        so the switch is a patched copy. Model paths inside resolve through
        `hf_hub_download`, not relative to this file, so its location does not matter.
        """
        source = repo / PIPELINE_CONFIGS[self.variant]
        try:
            config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise BackendError(f"cannot read LTX pipeline config {source}: {exc}") from exc

        config["prompt_enhancement_words_threshold"] = 0

        patched = tmp / source.name
        patched.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return patched

    @staticmethod
    def _newest_video(directory: Path, since: float) -> Path | None:
        """LTX names its own output file; find whatever it just wrote.

        Only files written by *this* run count. Rendering repeatedly into one output
        directory otherwise returns a previous clip when the current run writes nothing,
        pairing stale video with a fresh provenance sidecar — the exact error the
        sidecar exists to rule out. `since` is wall-clock, to match st_mtime.
        """
        candidates = [
            path
            for pattern in ("*.mp4", "*.webm")
            for path in directory.glob(pattern)
            if path.stat().st_mtime >= since
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
