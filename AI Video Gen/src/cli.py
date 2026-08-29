"""Command-line entry point.

    python -m src.cli "a red fox in a snowstorm"
    python -m src.cli "a red fox" --dry-run --enricher passthrough
    python -m src.cli "neon alley at night" --style cyberpunk --backend ltx --dialect ltx
    python -m src.cli --check
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import yaml

from .backends import BACKENDS, BackendError, get_backend
from .config import PROMPTS_DIR, OUTPUTS_DIR, GenerationRequest, Settings
from .enrich import ENRICHERS, EnrichmentError, get_enricher
from . import pipeline


def _styles() -> list[str]:
    data = yaml.safe_load((PROMPTS_DIR / "style_presets.yaml").read_text(encoding="utf-8"))
    return sorted(data or {})


def _dialects() -> list[str]:
    data = yaml.safe_load((PROMPTS_DIR / "model_dialects.yaml").read_text(encoding="utf-8"))
    return sorted(data or {})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-video-gen",
        description="Prompt -> enriched prompt -> video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", help="your short idea, in plain words")
    parser.add_argument("--enricher", default="ollama", choices=sorted(ENRICHERS),
                        help="stage-1 provider (default: ollama)")
    parser.add_argument("--backend", default="ltx", choices=sorted(BACKENDS),
                        help="stage-2 backend (default: ltx)")
    parser.add_argument("--style", default="cinematic", choices=_styles(),
                        help="visual style preset (default: cinematic)")
    parser.add_argument("--dialect", default="generic", choices=_dialects(),
                        help="phrasing tuned for the target model (default: generic)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=5.0, help="seconds (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="enrich only; skip generation. Use this to tune prompts cheaply.")
    parser.add_argument("--check", action="store_true",
                        help="report which enrichers and backends are ready, then exit")
    return parser


def run_check(settings: Settings) -> int:
    def report(name: str, component, error: str | None = None) -> None:
        if error is not None:
            print(f"  {name:12} BLOCKED  {error}")
            return
        problems, notes = component.preflight(), component.warnings()
        status = "BLOCKED " if problems else "ready   "
        print(f"  {name:12} {status} {component.model}")
        for line in problems + [f"(warning) {note}" for note in notes]:
            print(f"  {'':12}          {line}")

    print("Enrichers")
    for name in sorted(ENRICHERS):
        try:
            report(name, get_enricher(name, settings))
        except EnrichmentError as exc:
            report(name, None, str(exc))

    print("\nBackends")
    for name in sorted(BACKENDS):
        try:
            report(name, get_backend(name, settings))
        except BackendError as exc:
            report(name, None, str(exc))

    token = settings.hf_token
    if token:
        import hashlib

        digest = hashlib.sha256(token.encode()).hexdigest()[:8]
        source = "env" if os.environ.get("HF_TOKEN") == token else ".env/.env.local"
        print(f"\nHF_TOKEN  {token[:7]}... (sha {digest}, from {source})")
        print("  If you rotated this token and the fingerprint has not changed, the value "
              "is pinned\n  to this session. Start a new session, or put it in .env.local "
              "which overrides the\n  environment.")

    print(f"\nOutputs -> {OUTPUTS_DIR}")
    print("Nothing ready? `--enricher passthrough --backend stub` always works.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.check:
        return run_check(settings)
    if not args.prompt:
        build_parser().print_help()
        return 2

    request = GenerationRequest(
        prompt=args.prompt,
        style=args.style,
        dialect=args.dialect,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
    )

    try:
        result = pipeline.run(
            request,
            enricher_name=args.enricher,
            backend_name=args.backend,
            dry_run=args.dry_run,
            settings=settings,
            on_event=lambda message: print(f"[ai-video-gen] {message}", file=sys.stderr),
        )
    except (EnrichmentError, BackendError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: run `python -m src.cli --check` to see what is configured.", file=sys.stderr)
        return 1

    print(f"\nOriginal:\n  {result.prompt.original}\n")
    print("Enriched:")
    print(textwrap.fill(result.prompt.enriched, width=88,
                        initial_indent="  ", subsequent_indent="  "))
    print(f"\nFrames: {request.num_frames} @ {request.fps}fps "
          f"({request.width}x{request.height}, seed {request.seed})")
    if result.video_path:
        print(f"Video:  {result.video_path}")
    print(f"Record: {result.sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
