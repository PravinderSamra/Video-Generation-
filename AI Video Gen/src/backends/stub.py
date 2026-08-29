"""Stub backend — renders without any model, GPU, or network.

Its job is to prove the *pipeline* works end to end (enrichment, parameter resolution,
file naming, provenance sidecar) before you spend an hour downloading model weights.
Also what CI runs.

It draws real frames: a colour gradient that animates over time with the prompt's seed
mixed in, so two different prompts produce visibly different clips.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path

from ..config import EnrichedPrompt, GenerationRequest
from ..imageio import Image, write_png
from .base import BackendError, RenderResult, VideoBackend


def _frame_pixels(width: int, height: int, t: float, seed: int) -> bytes:
    """A cheap animated gradient. Deterministic in (t, seed)."""
    phase = (seed % 360) * math.pi / 180.0
    row = bytearray()
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            r = 0.5 + 0.5 * math.sin(6.0 * u + 2.0 * t + phase)
            g = 0.5 + 0.5 * math.sin(5.0 * v + 1.7 * t + phase + 2.1)
            b = 0.5 + 0.5 * math.sin(4.0 * (u + v) + 1.3 * t + phase + 4.2)
            row += bytes((int(r * 255), int(g * 255), int(b * 255)))
    return bytes(row)


class StubBackend(VideoBackend):
    name = "stub"
    model = "synthetic-gradient"

    def warnings(self) -> list[str]:
        if shutil.which("ffmpeg") is None:
            return ["ffmpeg not found — emitting a PNG frame sequence instead of an MP4"]
        return []

    def render(
        self,
        request: GenerationRequest,
        prompt: EnrichedPrompt,
        output_path: Path,
    ) -> RenderResult:
        started = time.monotonic()
        # Keep the stub fast: render small and let ffmpeg scale up.
        width, height = 128, max(1, int(128 * request.height / request.width))
        frames_dir = output_path.parent / f"{output_path.stem}-frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for index in range(request.num_frames):
            t = index / max(1, request.num_frames - 1) * 2.0 * math.pi
            write_png(
                frames_dir / f"frame_{index:04d}.png",
                Image(width, height, _frame_pixels(width, height, t, request.seed)),
            )

        notes: list[str] = ["synthetic output — no diffusion model was run"]
        if shutil.which("ffmpeg") is None:
            notes.append(f"ffmpeg unavailable; frames left in {frames_dir}")
            return RenderResult(
                video_path=frames_dir,
                backend=self.name,
                model=self.model,
                elapsed_seconds=time.monotonic() - started,
                notes=notes,
            )

        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(request.fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-vf", f"scale={request.width}:{request.height}:flags=neighbor",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise BackendError(f"ffmpeg failed: {result.stderr.strip()[:400]}")

        for frame in frames_dir.glob("frame_*.png"):
            frame.unlink()
        frames_dir.rmdir()

        return RenderResult(
            video_path=output_path,
            backend=self.name,
            model=self.model,
            elapsed_seconds=time.monotonic() - started,
            notes=notes,
        )
