"""Hugging Face backend — hosted generation, no local GPU required.

The no-GPU escape hatch, and the one path that works from a phone.

IMPORTANT, learned the hard way: no video model is served by the `hf-inference`
provider. Every one routes through a third party — fal-ai, replicate, wavespeed — and
each takes a differently shaped payload. Hand-rolling that HTTP is how you get a backend
that looks right and 404s. `huggingface_hub.InferenceClient` normalises provider
differences and picks a live provider for the model, so we use it rather than urllib.

Setup:
    pip install huggingface_hub
    free token at https://huggingface.co/settings/tokens
    set HF_TOKEN in the environment (not in a file you might commit)

Pick a model that is actually served — check with `list_served_video_models()` below, or
https://huggingface.co/models?pipeline_tag=text-to-video&inference_provider=all — and set
HF_VIDEO_MODEL. What is served changes over time; a plausible-looking model id that no
provider carries fails with a confusing error, so verify rather than assume.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from ..config import EnrichedPrompt, GenerationRequest, Settings
from .base import BackendError, RenderResult, VideoBackend

# Served at time of writing, cheapest first. Verify before trusting — see the docstring.
KNOWN_SERVED = (
    "Wan-AI/Wan2.1-T2V-1.3B",
    "Lightricks/LTX-Video-0.9.7-distilled",
    "Wan-AI/Wan2.2-TI2V-5B",
    "tencent/HunyuanVideo-1.5",
    "Wan-AI/Wan2.2-T2V-A14B",
)


# A fine-grained token defaults to repo access only. Without the inference permission
# every call 403s no matter how many credits the account has — and the provider's own
# error says "check your permissions" without naming which one. Catch it in preflight,
# using a free metadata call, before any credit is spent.
INFERENCE_PERMISSION_HINT = (
    "the token cannot call Inference Providers. At "
    "https://huggingface.co/settings/tokens either create a token of type 'Read', or "
    "edit this fine-grained token and tick Inference \u2192 "
    "'Make calls to Inference Providers'."
)


def token_can_call_inference(token: str) -> tuple[bool, str]:
    """Check the token carries an inference permission. Free — metadata only.

    Returns (ok, detail). A non-fine-grained ('read'/'write') token always qualifies;
    a fine-grained one has to name an inference permission explicitly.
    """
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
    except Exception as exc:
        return True, f"could not verify token permissions ({exc})"  # fail open, not closed

    access = (body.get("auth") or {}).get("accessToken") or {}
    role = access.get("role")
    if role != "fineGrained":
        return True, f"token role {role!r}"

    fine = access.get("fineGrained") or {}
    granted = set(fine.get("global") or [])
    for scope in fine.get("scoped") or []:
        granted.update(scope.get("permissions") or [])
    if any("inference" in permission for permission in granted):
        return True, "fine-grained token carries an inference permission"
    return False, f"fine-grained token grants only: {', '.join(sorted(granted)) or 'nothing'}"


def list_served_video_models(token: str, limit: int = 20) -> list[tuple[str, list[str]]]:
    """Which text-to-video models a provider currently serves. Free — metadata only."""
    url = (
        "https://huggingface.co/api/models?pipeline_tag=text-to-video"
        f"&inference_provider=all&limit={limit}&sort=likes&direction=-1"
        "&expand[]=inferenceProviderMapping"
    )
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        models = json.loads(response.read())
    return [
        (
            model["id"],
            [
                p["provider"]
                for p in (model.get("inferenceProviderMapping") or [])
                if p.get("status") == "live"
            ],
        )
        for model in models
    ]


class HFBackend(VideoBackend):
    name = "hf"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.hf_video_model

    def preflight(self) -> list[str]:
        problems = []
        if not self.settings.hf_token:
            problems.append(
                "HF_TOKEN is not set — create a free token at "
                "https://huggingface.co/settings/tokens"
            )
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            problems.append("huggingface_hub is not installed — `pip install huggingface_hub`")

        if self.settings.hf_token and not problems:
            ok, detail = token_can_call_inference(self.settings.hf_token)
            if not ok:
                problems.append(f"{INFERENCE_PERMISSION_HINT} ({detail})")
        return problems

    def warnings(self) -> list[str]:
        if self.model not in KNOWN_SERVED:
            return [
                f"{self.model} was not in the served list when this was written — "
                "confirm a provider carries it, or generation will fail confusingly"
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

        from huggingface_hub import InferenceClient

        client = InferenceClient(api_key=self.settings.hf_token)
        started = time.monotonic()
        try:
            video = client.text_to_video(
                prompt.enriched,
                model=self.model,
                negative_prompt=[prompt.negative],
                num_frames=float(request.num_frames),
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                seed=request.seed,
            )
        except Exception as exc:  # the client raises a wide range of provider errors
            message = str(exc)
            if "403" in message or "sufficient permissions" in message:
                raise BackendError(f"{INFERENCE_PERMISSION_HINT} Provider said: {message[:200]}") from exc
            if "402" in message or "credits" in message.lower():
                raise BackendError(
                    f"{self.model}: Inference Provider credits are exhausted for this "
                    f"account. Provider said: {message[:200]}"
                ) from exc
            raise BackendError(f"{self.model} via HF Inference Providers: {message}") from exc

        if not video or len(video) < 1024:
            raise BackendError(
                f"{self.model} returned {len(video or b'')} bytes — too small to be a video"
            )
        output_path.write_bytes(video)

        return RenderResult(
            video_path=output_path,
            backend=self.name,
            model=self.model,
            elapsed_seconds=time.monotonic() - started,
            notes=["hosted inference — consumes HF Inference Provider credits"],
            extra={"bytes": len(video)},
        )
