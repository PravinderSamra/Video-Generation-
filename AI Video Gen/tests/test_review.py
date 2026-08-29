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


def test_hear_pattern_does_not_match_heartbeat() -> None:
    """Regression: a loose `hear\\w*` matched "heartbeat" in real model output."""
    print("non-visual rule does not fire on visual words that merely start the same")
    for word, phrase in (
        ("heartbeat", "its heartbeat visible in the fur at its throat"),
        ("musician", "a street musician leaning against the wall"),
        ("heartland", "the open heartland stretching to the horizon"),
    ):
        violations = " ".join(lint_prompt(GOOD.replace("Muted palette", phrase + ", muted palette")).violations)
        check(f"{word} not flagged", "non-visual" not in violations)


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
        ("non-visual sense",
         GOOD.replace("Muted palette", "The scent of pine and a distant wail carry across, muted palette"),
         "non-visual sense"),
    ]
    for name, text, expected in cases:
        violations = " ".join(lint_prompt(text).violations)
        check(f"{name} detected", expected in violations)


def test_lint_catches_dropped_subject() -> None:
    print("linter advises (does not fail) on words the enrichment did not carry over")
    result = lint_prompt(GOOD, "a red fox and a golden eagle")
    advisories = " ".join(result.advisories)
    check("dropped 'eagle' reported", "eagle" in advisories)
    check("kept 'fox' not reported", "fox" not in advisories)
    check("reported as advisory, not violation", not result.violations)
    check("prompt still counts as ok", result.ok)


def test_advisory_does_not_fail_a_semantic_paraphrase() -> None:
    """Real HF output that paraphrased rather than repeated must not read as a failure.

    Qwen3-8B rendered "hunting in a snowstorm" as "paws at snowdrifts ... locks onto
    prey ... the storm's swirling snow". The idea survived; the words did not. That is
    an advisory for a human to judge, never a rule break.
    """
    print("a semantic paraphrase is advisory, not a violation")
    paraphrase = (
        "A lean red fox with a thick frost-tipped coat paws at deep snowdrifts on an open "
        "tundra, amber eye locking onto prey shivering beneath the white crust before it "
        "drops its shoulders to pounce. A 200mm telephoto holds a low medium shot, the "
        "storm's swirling snow compressed into a creamy grey wall behind it. Flat overcast "
        "light wraps the scene while ice crystals streak past the lens in slow motion. "
        "Rust orange, bone white, slate grey and pale blue, on 35mm film with fine grain "
        "and shallow depth of field."
    )
    result = lint_prompt(paraphrase, "a red fox hunting in a snowstorm")
    check(f"no violations (got {result.violations})", result.ok)
    check("flagged for human attention", bool(result.advisories))


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


def test_report_embeds_images_from_any_directory() -> None:
    """A report written outside the run directory must still embed its contact sheets.

    Regression: the embedder resolved sheets relative to the report's destination, so
    building into a scratch directory silently produced an image-less page.
    """
    print("report embeds contact sheets when built into another directory")
    import json as _json
    from src import pipeline
    from src.config import GenerationRequest
    from src.review import build_report, review_run

    with tempfile.TemporaryDirectory() as runs, tempfile.TemporaryDirectory() as elsewhere:
        pipeline.run(
            GenerationRequest("a red fox", duration=0.5, fps=16, width=64, height=64),
            enricher_name="passthrough", backend_name="stub", outputs_dir=Path(runs),
        )
        reviews = [review_run(s) for s in Path(runs).glob("*.json")]
        check("contact sheet produced", bool(reviews and reviews[0]["sheet"]))

        same = build_report(reviews, Path(runs) / "review.html").read_text()
        away = build_report(reviews, Path(elsewhere) / "review.html").read_text()
        check("embeds beside the runs", "data:image/png;base64," in same)
        check("embeds from elsewhere too", "data:image/png;base64," in away)

        fragment = build_report(
            reviews, Path(elsewhere) / "frag.html", standalone=False
        ).read_text()
        check("artifact build omits doctype", "<!doctype" not in fragment.lower())
        check("artifact build omits <head>", "<head>" not in fragment.lower())
        check("artifact build keeps the title", "<title>" in fragment)


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
        test_hear_pattern_does_not_match_heartbeat,
        test_lint_accepts_good_prompt,
        test_lint_catches_each_rule,
        test_lint_catches_dropped_subject,
        test_advisory_does_not_fail_a_semantic_paraphrase,
        test_healthy_clip_is_clean,
        test_detects_frozen,
        test_detects_flicker,
        test_detects_slideshow,
        test_detects_flat,
        test_detects_collapse,
        test_golden_set_is_lint_clean,
        test_report_embeds_images_from_any_directory,
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
