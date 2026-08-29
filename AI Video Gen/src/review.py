"""Review tooling — inspect what the pipeline produced.

Three things are being judged, and they need different instruments:

1. **Prompt quality** — `lint_prompt()` checks an enriched prompt against the rules the
   template itself states (length, one paragraph, no negations, no multi-shot). Cheap,
   deterministic, and catches the enricher drifting after a template edit.

2. **Video failure modes** — `frame_metrics()` detects the specific ways video diffusion
   fails: frozen output, slideshow stepping, flicker, collapse to mush. These are
   FAILURE DETECTORS, not quality scores. A clip can pass every check and still look bad;
   nothing here knows whether the fox looks like a fox.

3. **Aesthetic quality** — not automatable. `build_report()` produces a contact sheet and
   an HTML page so a human can compare runs side by side against a fixed scorecard.

Usage:
    python -m src.review                    # review everything in outputs/
    python -m src.review --lint-only        # prompts only, no frame extraction
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import OUTPUTS_DIR
from .imageio import Image, read_png, write_png

# ---------------------------------------------------------------------------
# 1. Prompt linting
# ---------------------------------------------------------------------------

MIN_WORDS, MAX_WORDS = 60, 110

# Words that signal the model negated instead of relying on the negative prompt.
NEGATIONS = re.compile(
    r"\b(no|not|without|never|avoid|avoiding|lacking|free of|absent|devoid)\b", re.I
)
# Instructions aimed at an operator rather than descriptions of the scene.
INSTRUCTIONS = re.compile(
    r"\b(please|make sure|ensure|should be|will be|we see|the viewer|render|generate|create a)\b",
    re.I,
)
# Markers of more than one shot. Diffusion models morph incoherently across cuts.
# "then" is deliberately narrow: mid-sentence it usually joins one continuous action
# ("coils and then dives"), which is fine. Only sentence-initial "Then" or "then,"
# reads as a transition. A linter that cries wolf gets ignored.
MULTI_SHOT = re.compile(
    r"(?:(?<=^)|(?<=\.\s))then\b|\bthen,|"
    r"\b(afterwards|meanwhile|cut to|next shot|moments later|seconds later|finally,)\b",
    re.I,
)
LEFTOVER_MARKUP = re.compile(r"(```|^[-*#>]\s|\*\*)", re.M)

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "and", "or", "with", "to", "for",
    "is", "are", "its", "it", "as", "by", "from", "into", "over", "under", "this",
}


@dataclass
class LintResult:
    violations: list[str] = field(default_factory=list)
    word_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations


def lint_prompt(enriched: str, original: str = "") -> LintResult:
    """Check an enriched prompt against the rules in prompts/cinematic_enrichment.md."""
    result = LintResult(word_count=len(enriched.split()))

    if result.word_count < MIN_WORDS:
        result.violations.append(
            f"too short: {result.word_count} words (min {MIN_WORDS}) — underspecified"
        )
    elif result.word_count > MAX_WORDS:
        result.violations.append(
            f"too long: {result.word_count} words (max {MAX_WORDS}) — detail gets diluted"
        )

    if "\n" in enriched.strip():
        result.violations.append("not a single paragraph — contains a line break")

    for label, pattern in (
        ("negation (put exclusions in the negative prompt)", NEGATIONS),
        ("operator instruction (describe the scene, not the task)", INSTRUCTIONS),
        ("multi-shot marker (one continuous shot only)", MULTI_SHOT),
        ("leftover markdown", LEFTOVER_MARKUP),
    ):
        found = pattern.findall(enriched)
        if found:
            unique = sorted({(f if isinstance(f, str) else f[0]).lower() for f in found})
            result.violations.append(f"{label}: {', '.join(unique[:4])}")

    if original:
        # Enrichment must build on the user's idea, not replace it.
        subject_words = {
            word for word in re.findall(r"[a-z]{3,}", original.lower())
            if word not in STOPWORDS
        }
        lowered = enriched.lower()
        dropped = sorted(w for w in subject_words if w[:5] not in lowered)
        if dropped:
            result.violations.append(f"dropped from the original prompt: {', '.join(dropped)}")

    return result


# ---------------------------------------------------------------------------
# 2. Video failure detection
# ---------------------------------------------------------------------------

# THRESHOLDS ARE PROVISIONAL. They were chosen against synthetic fixtures, not real
# renders — this project has never had a GPU. Calibrate them in Phase 1: run the
# benchmark set, record the metrics for clips you judge good and bad by eye, and move
# the constants so the flags agree with your judgement. Until then, treat a flag as
# "look at this one", never as a verdict.
FROZEN_DELTA = 0.004      # below this, consecutive frames are effectively identical
FLICKER_DELTA = 0.20      # above this on average, the clip is thrashing
SLIDESHOW_RATIO = 3.5     # max/mean delta ratio indicating stepped rather than fluid motion
FLAT_DETAIL = 0.03        # spatial standard deviation below this is a collapsed frame
PIXEL_STRIDE = 7          # sample every Nth pixel; these are heuristics, not measurements


@dataclass
class FrameMetrics:
    frame_count: int
    mean_delta: float
    max_delta: float
    frozen_frames: int
    mean_detail: float
    detail_drop: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "mean_delta": round(self.mean_delta, 5),
            "max_delta": round(self.max_delta, 5),
            "frozen_frames": self.frozen_frames,
            "mean_detail": round(self.mean_detail, 5),
            "detail_drop": round(self.detail_drop, 5),
            "flags": self.flags,
        }


def _luma_samples(image: Image) -> list[float]:
    pixels = image.pixels
    return [
        (pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114) / 255.0
        for i in range(0, len(pixels) - 2, 3 * PIXEL_STRIDE)
    ]


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def frame_metrics(frames: list[Image]) -> FrameMetrics:
    """Detect known video-diffusion failure modes. Not a quality score.

    These catch: a frozen clip, a slideshow, flicker, and collapse to flat mush.
    They cannot tell you whether the content is any good.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames to measure motion")

    lumas = [_luma_samples(frame) for frame in frames]
    details = [_stddev(luma) for luma in lumas]

    deltas = []
    for previous, current in zip(lumas, lumas[1:]):
        n = min(len(previous), len(current))
        deltas.append(sum(abs(previous[i] - current[i]) for i in range(n)) / max(1, n))

    mean_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    mean_detail = sum(details) / len(details)
    # Compare the first and last thirds: collapse shows up as detail draining away.
    third = max(1, len(details) // 3)
    head, tail = details[:third], details[-third:]
    detail_drop = (sum(head) / len(head)) - (sum(tail) / len(tail))

    flags: list[str] = []
    if mean_delta < FROZEN_DELTA:
        flags.append("FROZEN — almost no motion between frames")
    if mean_delta > FLICKER_DELTA:
        flags.append("FLICKER — frame-to-frame change is extreme")
    if mean_delta > 0 and max_delta / mean_delta > SLIDESHOW_RATIO:
        flags.append("SLIDESHOW — motion is stepped rather than continuous")
    if mean_detail < FLAT_DETAIL:
        flags.append("FLAT — frames carry very little spatial detail")
    if detail_drop > 0.05:
        flags.append("COLLAPSE — detail drains away toward the end of the clip")

    return FrameMetrics(
        frame_count=len(frames),
        mean_delta=mean_delta,
        max_delta=max_delta,
        frozen_frames=sum(1 for d in deltas if d < FROZEN_DELTA),
        mean_detail=mean_detail,
        detail_drop=detail_drop,
        flags=flags,
    )


def extract_frames(video: Path, destination: Path, limit: int = 24) -> list[Image]:
    """Pull frames from a video (via ffmpeg) or read a stub frame directory directly."""
    if video.is_dir():  # stub backend output when ffmpeg is unavailable
        paths = sorted(video.glob("frame_*.png"))
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to extract frames from a video file")
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-pix_fmt", "rgb24", str(destination / "frame_%04d.png")],
            check=True, capture_output=True,
        )
        paths = sorted(destination.glob("frame_*.png"))

    if not paths:
        raise RuntimeError(f"no frames found for {video}")
    step = max(1, len(paths) // limit)
    return [read_png(path) for path in paths[::step][:limit]]


# ---------------------------------------------------------------------------
# 3. Human review: contact sheet + HTML report
# ---------------------------------------------------------------------------

def contact_sheet(frames: list[Image], columns: int = 6, cell_width: int = 160) -> Image:
    """Tile sampled frames into one grid image, so a whole clip is legible at a glance."""
    if not frames:
        raise ValueError("no frames to tile")
    aspect = frames[0].height / frames[0].width
    cell_height = max(1, int(cell_width * aspect))
    columns = min(columns, len(frames))
    rows = (len(frames) + columns - 1) // columns
    gap = 4

    sheet_width = columns * cell_width + (columns + 1) * gap
    sheet_height = rows * cell_height + (rows + 1) * gap
    canvas = bytearray(b"\x18" * (sheet_width * sheet_height * 3))

    for index, frame in enumerate(frames):
        thumb = frame.resized(cell_width, cell_height)
        row, column = divmod(index, columns)
        x0 = gap + column * (cell_width + gap)
        y0 = gap + row * (cell_height + gap)
        for y in range(cell_height):
            src = y * cell_width * 3
            dst = ((y0 + y) * sheet_width + x0) * 3
            canvas[dst : dst + cell_width * 3] = thumb.pixels[src : src + cell_width * 3]

    return Image(sheet_width, sheet_height, bytes(canvas))


SCORECARD = [
    ("Prompt adherence", "Is everything the prompt asked for actually present?"),
    ("Subject coherence", "Does the subject hold its shape and identity across all frames?"),
    ("Motion quality", "Is the movement fluid and physically plausible, not stepped or floaty?"),
    ("Temporal stability", "Any flicker, morphing, or background churn?"),
    ("Cinematography", "Did the requested shot size, camera move, and lighting land?"),
    ("Style match", "Does it read as the chosen style preset?"),
]


def review_run(sidecar: Path, extract: bool = True) -> dict[str, Any]:
    """Review one pipeline run from its provenance sidecar."""
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    enrichment = record.get("enrichment", {})

    lint = lint_prompt(enrichment.get("enriched", ""), enrichment.get("original", ""))
    review: dict[str, Any] = {
        "name": sidecar.stem,
        "record": record,
        "lint": {"ok": lint.ok, "violations": lint.violations, "word_count": lint.word_count},
        "metrics": None,
        "sheet": None,
        "error": None,
    }

    video = record.get("video_path")
    if not (extract and video and Path(video).exists()):
        return review

    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = extract_frames(Path(video), Path(tmp))
        review["metrics"] = frame_metrics(frames).to_dict()
        sheet_path = sidecar.with_name(f"{sidecar.stem}-sheet.png")
        write_png(sheet_path, contact_sheet(frames))
        review["sheet"] = sheet_path.name
    except Exception as exc:  # a bad clip must not abort the whole report
        review["error"] = f"{type(exc).__name__}: {exc}"

    return review


def build_report(reviews: list[dict[str, Any]], destination: Path) -> Path:
    """Write a self-contained HTML page for side-by-side human review."""
    blocks = []
    for review in reviews:
        record = review["record"]
        enrichment = record.get("enrichment", {})
        request = record.get("request", {})
        render = record.get("render") or {}
        lint = review["lint"]
        metrics = review["metrics"]

        badges = [
            f'<span class="badge {"ok" if lint["ok"] else "bad"}">'
            f'prompt {"clean" if lint["ok"] else str(len(lint["violations"])) + " issues"}</span>'
        ]
        if metrics:
            flags = metrics["flags"]
            badges.append(
                f'<span class="badge {"ok" if not flags else "bad"}">'
                f'video {"clean" if not flags else str(len(flags)) + " flags"}</span>'
            )

        problems = "".join(f"<li>{html.escape(v)}</li>" for v in lint["violations"])
        problems += "".join(
            f"<li>{html.escape(f)}</li>" for f in (metrics["flags"] if metrics else [])
        )
        if review["error"]:
            problems += f"<li>{html.escape(review['error'])}</li>"

        stats = ""
        if metrics:
            stats = (
                f'<div class="stats">frames {metrics["frame_count"]} · '
                f'motion {metrics["mean_delta"]:.4f} · peak {metrics["max_delta"]:.4f} · '
                f'detail {metrics["mean_detail"]:.4f} · frozen {metrics["frozen_frames"]}</div>'
            )

        sheet = (
            f'<img src="{html.escape(review["sheet"])}" alt="contact sheet">'
            if review["sheet"] else '<p class="muted">no contact sheet — video not rendered</p>'
        )
        scorecard = "".join(
            f"<tr><td>{html.escape(name)}</td><td class='muted'>{html.escape(hint)}</td>"
            f"<td class='score'>&nbsp;/5</td></tr>"
            for name, hint in SCORECARD
        )

        blocks.append(f"""
<section>
  <h2>{html.escape(review["name"])} {"".join(badges)}</h2>
  <div class="meta">
    {html.escape(str(enrichment.get("provider")))} &rarr;
    {html.escape(str(render.get("backend", "not rendered")))}
    &nbsp;·&nbsp; {html.escape(str(request.get("style")))}
    &nbsp;·&nbsp; seed {html.escape(str(request.get("seed")))}
    &nbsp;·&nbsp; {html.escape(str(request.get("width")))}&times;{html.escape(str(request.get("height")))}
    &nbsp;·&nbsp; {html.escape(str(request.get("num_frames")))} frames
  </div>
  <p class="original"><strong>Original:</strong> {html.escape(enrichment.get("original", ""))}</p>
  <p class="enriched">{html.escape(enrichment.get("enriched", ""))}
     <span class="muted">({lint["word_count"]} words)</span></p>
  {f'<ul class="problems">{problems}</ul>' if problems else ''}
  {stats}
  {sheet}
  <table class="scorecard"><tbody>{scorecard}</tbody></table>
</section>""")

    clean = sum(1 for r in reviews if r["lint"]["ok"] and not (r["metrics"] or {}).get("flags"))
    page = f"""<!doctype html>
<meta charset="utf-8"><title>AI Video Gen — review</title>
<style>
 body{{font:15px/1.6 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1.5rem;
       color:#1a1a1a;background:#fafafa}}
 h1{{margin-bottom:.2rem}} h2{{font-size:1.05rem;margin:0 0 .4rem}}
 section{{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:1.2rem;margin:1.2rem 0}}
 .meta{{font-size:.85rem;color:#666;margin-bottom:.8rem}}
 .original{{color:#555;font-size:.9rem}} .enriched{{background:#f4f7fb;padding:.7rem .9rem;border-radius:6px}}
 .muted{{color:#888;font-weight:400}}
 .badge{{font-size:.7rem;padding:.15rem .5rem;border-radius:99px;margin-left:.5rem;vertical-align:middle}}
 .badge.ok{{background:#e3f5e8;color:#1c6b33}} .badge.bad{{background:#fdeaea;color:#a12626}}
 .problems{{background:#fdf6f6;border-left:3px solid #d97070;padding:.6rem 1.2rem;margin:.8rem 0;
            font-size:.88rem;color:#8a2b2b}}
 .stats{{font-family:ui-monospace,monospace;font-size:.8rem;color:#555;margin:.6rem 0}}
 img{{max-width:100%;border-radius:6px;border:1px solid #ddd;margin-top:.5rem}}
 .scorecard{{width:100%;border-collapse:collapse;margin-top:1rem;font-size:.85rem}}
 .scorecard td{{border-top:1px solid #eee;padding:.4rem .3rem}}
 .score{{text-align:right;color:#aaa;font-family:ui-monospace,monospace;width:4rem}}
</style>
<h1>AI Video Gen — review</h1>
<p class="muted">{len(reviews)} run(s) · {clean} with no automated findings.
Automated checks catch failure modes only; the scorecard is for the part that needs eyes.</p>
{''.join(blocks)}
"""
    destination.write_text(page, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="review", description="Review pipeline output.")
    parser.add_argument("directory", nargs="?", default=str(OUTPUTS_DIR))
    parser.add_argument("--lint-only", action="store_true",
                        help="check prompts only; skip frame extraction")
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    sidecars = sorted(p for p in directory.glob("*.json"))
    if not sidecars:
        print(f"no runs found in {directory} — generate something first")
        return 1

    reviews = [review_run(sidecar, extract=not args.lint_only) for sidecar in sidecars]

    for review in reviews:
        flags = (review["metrics"] or {}).get("flags", [])
        problems = review["lint"]["violations"] + flags
        status = "ok" if not problems else f"{len(problems)} finding(s)"
        print(f"{review['name']:52} {status}")
        for problem in problems:
            print(f"  - {problem}")

    report = build_report(reviews, directory / "review.html")
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
