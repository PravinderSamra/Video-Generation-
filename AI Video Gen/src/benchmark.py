"""Run the fixed benchmark set from prompts/benchmark.yaml.

    python -m src.benchmark --dry-run              # enrichment only — free, seconds
    python -m src.benchmark --backend ltx          # full render
    python -m src.benchmark --only fox_snow,candle_still

Always at the same seeds, so a change to the enrichment template or a backend swap
produces a comparable set rather than an anecdote. Follow with `python -m src.review`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .backends import BACKENDS, BackendError
from .config import PROMPTS_DIR, OUTPUTS_DIR, GenerationRequest, Settings
from .enrich import ENRICHERS, EnrichmentError
from . import pipeline


def load_benchmark() -> list[dict]:
    data = yaml.safe_load((PROMPTS_DIR / "benchmark.yaml").read_text(encoding="utf-8"))
    return data.get("prompts", [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__)
    parser.add_argument("--enricher", default="ollama", choices=sorted(ENRICHERS))
    parser.add_argument("--backend", default="stub", choices=sorted(BACKENDS))
    parser.add_argument("--dry-run", action="store_true", help="enrich only, skip rendering")
    parser.add_argument("--only", help="comma-separated benchmark ids to run")
    parser.add_argument("--outputs", default=str(OUTPUTS_DIR / "benchmark"),
                        help="directory for results (default: outputs/benchmark)")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    args = parser.parse_args(argv)

    entries = load_benchmark()
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        entries = [entry for entry in entries if entry["id"] in wanted]
        if not entries:
            print(f"no benchmark entries matched {sorted(wanted)}", file=sys.stderr)
            return 2

    outputs = Path(args.outputs)
    settings = Settings.from_env()
    failures = 0

    for index, entry in enumerate(entries, 1):
        print(f"[{index}/{len(entries)}] {entry['id']}: {entry['prompt']}")
        request = GenerationRequest(
            prompt=entry["prompt"],
            style=entry.get("style", "cinematic"),
            dialect=entry.get("dialect", "generic"),
            seed=entry["seed"],
            duration=args.duration,
            width=args.width,
            height=args.height,
        )
        try:
            pipeline.run(
                request,
                enricher_name=args.enricher,
                backend_name=args.backend,
                dry_run=args.dry_run,
                outputs_dir=outputs,
                settings=settings,
                on_event=lambda message: None,
            )
            print("      ok")
        except (EnrichmentError, BackendError) as exc:
            failures += 1
            print(f"      FAILED: {exc}")

    print(f"\n{len(entries) - failures}/{len(entries)} succeeded -> {outputs}")
    print(f"Now review: python -m src.review {outputs}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
