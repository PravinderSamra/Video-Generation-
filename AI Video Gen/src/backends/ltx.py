"""LTX-Video backend — local generation via the upstream LTX inference entrypoint.

Default backend: best quality-per-VRAM-per-second of the options evaluated, Apache-2.0,
and the only one with a clean documented single-command CLI.

Setup:
    git clone https://github.com/Lightricks/LTX-Video && cd LTX-Video
    python -m pip install -e .[inference]
    echo "LTX_REPO=$(pwd)" >> "<project>/.env"

We shell out rather than importing, so upstream refactors cannot break us. The
subprocess runs LAUNCHER, a shim that imports LTX in its own environment; the
clone itself is never modified.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from ..config import EnrichedPrompt, GenerationRequest, Settings
from .base import BackendError, RenderResult, VideoBackend

# Run LTX through this rather than its inference.py: upstream's pipeline construction
# does not fit a 16 GB card, and the fix is placement, not settings. See the docstring.
LAUNCHER = '''"""Run LTX inference, placing the models so a 16 GB card can hold them.

`create_ltx_video_pipeline` moves the transformer, VAE and text encoder onto the GPU
together, before anything can offload them. Measured on a T4 for 2b-distilled that is
3.58 + 4.65 + 8.87 = 17.10 GB against 14.56 GB usable, so it OOMs while loading, and no
choice of variant or resolution helps -- the models are the size they are. Two of those
figures are also inflated: upstream loads the VAE and text encoder in float32 and casts
both to bfloat16 immediately *after* moving them, so the float32 copies are transients
that only ever needed to exist on the CPU.

None of that eager placement is necessary. The pipeline already moves the text encoder
in before encoding and out after it, and the transformer in for denoising and out again.
So we build on the CPU and let it do that, which leaves a peak around 11.8 GB.

The VAE is the exception, pinned to the GPU throughout: its normalisation statistics are
read as plain buffers outside forward(), so nothing moves them on demand, and at 2.3 GB
in bfloat16 it is the one model cheap enough to keep resident.

accelerate's enable_model_cpu_offload() looks like the supported way to do this and is
not: maybe_free_model_hooks() re-runs it at the end of every inner pass, re-hooking the
VAE and pulling it back to the CPU mid-render.

This lives here rather than in the clone, so LTX stays a pristine checkout we shell into
and an upstream fix turns the shim into a no-op rather than a conflict.
"""
import torch
from transformers import HfArgumentParser

import ltx_video.inference as ltx
import ltx_video.pipelines.pipeline_ltx_video as pipeline_module

DEVICE = ltx.get_device()

if DEVICE != "cpu":
    # Load at the dtype upstream casts to anyway; the float32 copies stay off the card.
    _text_encoder = ltx.T5EncoderModel.from_pretrained
    ltx.T5EncoderModel.from_pretrained = lambda *a, **k: _text_encoder(
        *a, **{**k, "torch_dtype": torch.bfloat16}
    )
    _autoencoder = ltx.CausalVideoAutoencoder.from_pretrained
    ltx.CausalVideoAutoencoder.from_pretrained = lambda *a, **k: _autoencoder(*a, **k).to(
        torch.bfloat16
    )

    _create_pipeline = ltx.create_ltx_video_pipeline

    def _create_on_cpu(*args, **kwargs):
        kwargs["device"] = "cpu"
        pipeline = _create_pipeline(*args, **kwargs)
        pipeline.vae.to(DEVICE)
        return pipeline

    ltx.create_ltx_video_pipeline = _create_on_cpu

    # The models now genuinely straddle two devices, so the inherited property would
    # report whichever one it finds first. State it instead.
    pipeline_module.LTXVideoPipeline._execution_device = property(
        lambda self: torch.device(DEVICE)
    )

    # Built from pipeline.device upstream, which is now the CPU, while the latents it
    # is asserted against are produced on DEVICE.
    _create_upsampler = ltx.create_latent_upsampler
    ltx.create_latent_upsampler = lambda path, device: _create_upsampler(path, DEVICE)

ltx.infer(config=HfArgumentParser(ltx.InferenceConfig).parse_args_into_dataclasses()[0])
'''

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
            launcher = Path(tmp) / "run_ltx.py"
            launcher.write_text(LAUNCHER, encoding="utf-8")
            command = [
                "python", str(launcher),
                "--prompt", prompt.enriched,
                "--negative_prompt", prompt.negative,
                "--height", str(request.height),
                "--width", str(request.width),
                "--num_frames", str(request.num_frames),
                "--frame_rate", str(request.fps),
                "--seed", str(request.seed),
                "--pipeline_config", str(self._pipeline_config(repo, Path(tmp))),
                "--output_path", str(output_path.parent),
                # Load-bearing alongside LAUNCHER, not a nicety: it is what evicts the
                # text encoder once it has encoded. Leaving it resident through denoising
                # is 8.87 + 3.58 + 2.33 = 14.78 GB, over a T4 again. Upstream self-
                # disables it above 30 GB of VRAM, so a large card pays nothing for it.
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
                f"LTX inference exited {result.returncode}:\n{result.stderr.strip()[-1500:]}"
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
        transformer. Ours are always shorter than the shipped threshold of 120, so it
        always fires.

        This is a provenance fix, not a memory one — the enhancer is loaded after the
        point a 16 GB card runs out of VRAM, so it is LAUNCHER, not this, that makes
        the render fit. What it costs us is the sidecar: stage one already produced
        `prompt.enriched`, and the sidecar records it as what was rendered. Letting LTX
        rewrite the prompt after that makes the sidecar describe something other than
        the clip beside it, which is the one claim the sidecar exists to make.

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
