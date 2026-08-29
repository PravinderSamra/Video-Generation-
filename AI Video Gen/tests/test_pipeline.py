"""Smoke tests. No GPU, no network, no model weights — these must pass anywhere.

    python -m tests.test_pipeline
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends import BACKENDS, get_backend           # noqa: E402
from src.config import GenerationRequest, Settings, load_style, load_system_prompt  # noqa: E402
from src.enrich import ENRICHERS, build_system_prompt    # noqa: E402
from src import pipeline                                 # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        FAILURES.append(label)


def test_frame_snapping() -> None:
    print("frame count snaps to 8n+1 (VAE requirement)")
    for duration, fps in ((5.0, 24), (1.0, 24), (3.0, 30), (0.1, 8)):
        frames = GenerationRequest("x", duration=duration, fps=fps).num_frames
        check(f"{duration}s @ {fps}fps -> {frames}", frames % 8 == 1 and frames >= 9)


def test_template_substitution() -> None:
    print("enrichment template has no unsubstituted placeholders")
    request = GenerationRequest("a fox", style="noir", dialect="ltx", duration=5)
    rendered = build_system_prompt(request)
    check("no {{...}} left", "{{" not in rendered)
    check("style injected", "noir" in rendered.lower())
    check("dialect injected", "ltx-video" in rendered.lower())
    check("duration injected", "5 seconds" in rendered)


def test_registries_construct() -> None:
    print("every registered component constructs and reports readiness")
    settings = Settings.from_env()
    for name in BACKENDS:
        backend = get_backend(name, settings)
        check(f"backend {name}", isinstance(backend.preflight(), list))
    check("enrichers registered", set(ENRICHERS) == {"ollama", "hf", "passthrough", "fixture"})


def test_end_to_end() -> None:
    print("full pipeline runs with no external dependencies")
    with tempfile.TemporaryDirectory() as tmp:
        outputs = Path(tmp)
        result = pipeline.run(
            GenerationRequest("a red fox", duration=0.5, fps=16, width=64, height=64, seed=3),
            enricher_name="passthrough",
            backend_name="stub",
            outputs_dir=outputs,
        )
        check("sidecar written", result.sidecar_path.is_file())
        check("output produced", result.video_path is not None and result.video_path.exists())
        check("provenance recorded", result.to_dict()["request"]["seed"] == 3)

        # A second identical run must not clobber the first.
        again = pipeline.run(
            GenerationRequest("a red fox", duration=0.5, fps=16, width=64, height=64, seed=3),
            enricher_name="passthrough",
            backend_name="stub",
            outputs_dir=outputs,
        )
        check("no filename collision", again.sidecar_path != result.sidecar_path)


def test_dry_run_skips_generation() -> None:
    print("dry run stops after enrichment")
    with tempfile.TemporaryDirectory() as tmp:
        result = pipeline.run(
            GenerationRequest("a red fox"),
            enricher_name="passthrough",
            backend_name="stub",
            dry_run=True,
            outputs_dir=Path(tmp),
        )
        check("no video rendered", result.video_path is None)
        check("prompt still cached", result.sidecar_path.is_file())


def main() -> int:
    for test in (
        test_frame_snapping,
        test_template_substitution,
        test_registries_construct,
        test_end_to_end,
        test_dry_run_skips_generation,
    ):
        test()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
