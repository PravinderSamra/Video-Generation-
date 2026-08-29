"""Tests for the review tooling.

The important ones are the failure-mode tests: a detector that never fires is worse
than no detector, because it manufactures false confidence. Each synthesises the exact
failure it claims to catch and asserts the right flag appears.

    python -m tests.test_review
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.imageio import Image                    # noqa: E402
from src.review import (                          # noqa: E402
    contact_sheet,
    frame_metrics,
    lint_prompt,
)

FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        FAILURES.append(label)


# --- frame synthesis helpers ------------------------------------------------

W = H = 24


def noisy(seed: int, scale: float = 1.0, base: int = 128) -> Image:
    rng = random.Random(seed)
    return Image(W, H, bytes(
        max(0, min(255, int(base + rng.uniform(-127, 127) * scale)))
        for _ in range(W * H * 3)
    ))


def drifting(index: int) -> Image:
    """A gently moving gradient — what a healthy clip looks like."""
    px = bytearray()
    for y in range(H):
        for x in range(W):
            v = 0.5 + 0.5 * math.sin((x + index * 0.6) * 0.4 + y * 0.2)
            px += bytes((int(v * 255), int(v * 200), int(v * 150)))
    return Image(W, H, bytes(px))


# --- prompt linting ---------------------------------------------------------

GOOD = (
    "A lean red fox with a thick frost-tipped winter coat pushes through deep powder, "
    "ears flicking as it locks onto something beneath the surface, then coils and dives "
    "nose-first into the drift. The camera holds a low telephoto medium shot at 200mm, "
    "compressing the blizzard into a soft grey wall behind it. Overcast daylight wraps "
    "the scene in flat diffuse light, snow crystals drifting across the frame in slow "
    "motion. Muted palette of rust orange, bone white, slate grey and pale blue, captured "
    "on 35mm film with fine natural grain and shallow depth of field throughout the shot."
)


def test_lint_accepts_good_prompt() -> None:
    print("linter accepts a well-formed enriched prompt")
    result = lint_prompt(GOOD, "a red fox in the snow")
    check(f"no violations (got {result.violations})", result.ok)
    check("word count in range", 60 <= result.word_count <= 110)


def test_lint_catches_each_rule() -> None:
    print("linter catches each rule violation it claims to")
    cases = [
        ("too short", "A fox runs.", "too short"),
        ("too long", GOOD + " " + GOOD, "too long"),
        ("negation", GOOD.replace("Muted palette", "No people visible, muted palette"), "negation"),
        ("instruction", GOOD.replace("The camera holds", "Please make sure the camera holds"),
         "operator instruction"),
        ("multi-shot (meanwhile)",
         GOOD.replace("Overcast daylight", "Meanwhile overcast daylight"), "multi-shot"),
        ("multi-shot (sentence-initial then)",
         GOOD.replace("Overcast daylight", "Then, overcast daylight"), "multi-shot"),
        ("multi-shot (cut to)",
         GOOD.replace("Overcast daylight", "Cut to overcast daylight"), "multi-shot"),
        ("line break", GOOD.replace("The camera", "\nThe camera"), "single paragraph"),
        ("markdown", "```\n" + GOOD + "\n```", "markdown"),
    ]
    for name, text, expected in cases:
        violations = " ".join(lint_prompt(text).violations)
        check(f"{name} detected", expected in violations)


def test_lint_catches_dropped_subject() -> None:
    print("linter catches enrichment that abandons the user's idea")
    violations = " ".join(lint_prompt(GOOD, "a red fox and a golden eagle").violations)
    check("dropped 'eagle' reported", "eagle" in violations)
    check("kept 'fox' not reported", "fox" not in violations)


# --- video failure detection ------------------------------------------------

def test_healthy_clip_is_clean() -> None:
    print("a healthy clip raises no flags")
    metrics = frame_metrics([drifting(i) for i in range(16)])
    check(f"no flags (got {metrics.flags})", not metrics.flags)
    check("motion detected", metrics.mean_delta > 0.004)


def test_detects_frozen() -> None:
    print("FROZEN fires on a clip with no motion")
    still = drifting(0)
    metrics = frame_metrics([still] * 12)
    check("FROZEN flagged", any(f.startswith("FROZEN") for f in metrics.flags))
    check("all deltas counted frozen", metrics.frozen_frames == 11)


def test_detects_flicker() -> None:
    print("FLICKER fires on frame-to-frame chaos")
    # Real flicker is content oscillating between states, not per-pixel white noise
    # (which averages away into luma). Alternate two very different frames.
    dark = Image(W, H, bytes([20] * (W * H * 3)))
    metrics = frame_metrics([drifting(i) if i % 2 else dark for i in range(12)])
    check("FLICKER flagged", any(f.startswith("FLICKER") for f in metrics.flags))
    check("healthy clip stays below the threshold",
          frame_metrics([drifting(i) for i in range(16)]).mean_delta < 0.20)


def test_detects_slideshow() -> None:
    print("SLIDESHOW fires on stepped rather than continuous motion")
    frames: list[Image] = []
    for step in range(4):          # hold four frames, then jump
        frames += [drifting(step * 9)] * 4
    metrics = frame_metrics(frames)
    check("SLIDESHOW flagged", any(f.startswith("SLIDESHOW") for f in metrics.flags))


def test_detects_flat() -> None:
    print("FLAT fires on frames with almost no spatial detail")
    frames = [Image(W, H, bytes([120 + i % 2] * (W * H * 3))) for i in range(10)]
    metrics = frame_metrics(frames)
    check("FLAT flagged", any(f.startswith("FLAT") for f in metrics.flags))


def test_detects_collapse() -> None:
    print("COLLAPSE fires when detail drains away over the clip")
    frames = []
    for i in range(15):
        fade = 1.0 - i / 15.0          # detail decays toward a flat frame
        frames.append(noisy(i, scale=fade))
    metrics = frame_metrics(frames)
    check("COLLAPSE flagged", any(f.startswith("COLLAPSE") for f in metrics.flags))
    check("detail drop positive", metrics.detail_drop > 0)


def test_golden_set_is_lint_clean() -> None:
    """The reference enrichments must obey the template they were written to.

    This is the guard that keeps prompts/golden_enrichments.yaml honest: if a rule in
    the linter changes, or someone edits a fixture, the mismatch surfaces here rather
    than silently degrading the baseline that real enricher output is judged against.
    """
    print("golden enrichment set passes its own linter")
    from src.enrich import FixtureEnricher
    import yaml as _yaml
    from src.config import PROMPTS_DIR

    entries = FixtureEnricher()._load()
    benchmark = _yaml.safe_load(
        (PROMPTS_DIR / "benchmark.yaml").read_text(encoding="utf-8")
    )["prompts"]

    check(f"one fixture per benchmark prompt ({len(entries)}/{len(benchmark)})",
          len(entries) == len(benchmark))
    for entry in benchmark:
        check(f"{entry['id']} has a fixture", entry["prompt"].strip().lower() in entries)

    for original, enriched in entries.items():
        result = lint_prompt(enriched, original)
        check(f"{original[:38]!r} lints clean", result.ok)
        for violation in result.violations:
            print(f"        - {violation}")


def test_contact_sheet_geometry() -> None:
    print("contact sheet tiles frames into the expected grid")
    sheet = contact_sheet([drifting(i) for i in range(7)], columns=3, cell_width=20)
    check("3 columns wide", sheet.width == 3 * 20 + 4 * 4)
    check("3 rows tall", sheet.height == 3 * 20 + 4 * 4)
    check("pixel buffer consistent", len(sheet.pixels) == sheet.width * sheet.height * 3)
    with tempfile.TemporaryDirectory() as tmp:
        from src.imageio import read_png, write_png
        path = Path(tmp) / "sheet.png"
        write_png(path, sheet)
        check("round-trips through PNG", read_png(path).pixels == sheet.pixels)


def main() -> int:
    for test in (
        test_lint_accepts_good_prompt,
        test_lint_catches_each_rule,
        test_lint_catches_dropped_subject,
        test_healthy_clip_is_clean,
        test_detects_frozen,
        test_detects_flicker,
        test_detects_slideshow,
        test_detects_flat,
        test_detects_collapse,
        test_golden_set_is_lint_clean,
        test_contact_sheet_geometry,
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
