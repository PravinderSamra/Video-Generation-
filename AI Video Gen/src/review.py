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

import base64
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
from .imageio import Image, encode_png, read_png, write_png

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


# Artifact pages cap at 16 MB, and a phone on mobile data has opinions too. Budget the
# whole embedded page well under that and divide it across however many runs there are.
EMBED_BUDGET_BYTES = 11_000_000
MIN_SHEET_BYTES = 120_000


def _embed_sheet(path: Path, budget: int) -> str:
    """Return a data URI for a contact sheet, downscaling until it fits the budget."""
    data = path.read_bytes()
    if len(data) <= budget:
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    image = read_png(path)
    for _ in range(6):  # halve the linear dimensions until it fits
        image = image.resized(max(64, image.width // 2), max(40, image.height // 2))
        data = encode_png(image)
        if len(data) <= budget or image.width <= 64:
            break
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


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
        "sheet": None,        # filename, for relative linking beside the sidecar
        "sheet_path": None,   # absolute, so a report built elsewhere can still embed
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
        review["sheet_path"] = str(sheet_path)
    except Exception as exc:  # a bad clip must not abort the whole report
        review["error"] = f"{type(exc).__name__}: {exc}"

    return review


PAGE_STYLES = """
<style>
:root{
  --ground:#f4f3f0; --surface:#fbfaf8; --surface-2:#edebe6;
  --ink:#16181c; --ink-2:#5c5f66; --ink-3:#8b8e95; --line:#dcd9d3;
  --accent:#b3701a; --pass:#2f6f4a; --flag:#a63a26;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#131417; --surface:#1a1c20; --surface-2:#23262b;
    --ink:#e8e6e2; --ink-2:#a2a5ac; --ink-3:#71757d; --line:#2c3037;
    --accent:#d99a3e; --pass:#5aa87a; --flag:#d4705c;
  }
}
:root[data-theme="dark"]{
  --ground:#131417; --surface:#1a1c20; --surface-2:#23262b;
  --ink:#e8e6e2; --ink-2:#a2a5ac; --ink-3:#71757d; --line:#2c3037;
  --accent:#d99a3e; --pass:#5aa87a; --flag:#d4705c;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:Archivo,"Helvetica Neue",Arial,sans-serif;
  font-size:15px; line-height:1.55; -webkit-text-size-adjust:100%;
}
.wrap{max-width:880px; margin:0 auto; padding:2rem 1.1rem 4rem; display:flex;
      flex-direction:column; gap:1.1rem}
header{border-bottom:2px solid var(--ink); padding-bottom:.9rem; margin-bottom:.4rem}
h1{font-size:1.55rem; font-weight:700; letter-spacing:-.02em; margin:0 0 .3rem;
   text-wrap:balance}
.lede{color:var(--ink-2); font-size:.9rem; margin:0; max-width:60ch}
.tally{font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.75rem;
       color:var(--ink-3); margin-top:.55rem; font-variant-numeric:tabular-nums}
.take{background:var(--surface); border:1px solid var(--line); border-radius:4px;
      overflow:hidden}
.slate{display:flex; flex-wrap:wrap; align-items:baseline; gap:.5rem .9rem;
       background:var(--surface-2); border-bottom:1px solid var(--line);
       padding:.6rem .95rem; font-family:"JetBrains Mono",ui-monospace,monospace;
       font-size:.72rem; color:var(--ink-2); font-variant-numeric:tabular-nums}
.slate .id{color:var(--ink); font-weight:700; letter-spacing:.02em}
.slate .sep{color:var(--ink-3)}
.status{margin-left:auto; display:flex; gap:.4rem}
.chip{font-size:.65rem; letter-spacing:.06em; text-transform:uppercase;
      padding:.15rem .45rem; border-radius:2px; font-weight:700}
.chip.pass{background:color-mix(in srgb,var(--pass) 16%,transparent); color:var(--pass)}
.chip.flag{background:color-mix(in srgb,var(--flag) 16%,transparent); color:var(--flag)}
.body{padding:1rem .95rem 1.1rem; display:flex; flex-direction:column; gap:.85rem}
.in{font-size:.8rem; color:var(--ink-3); font-family:"JetBrains Mono",ui-monospace,monospace}
.in b{color:var(--ink-2); font-weight:500}
.out{font-family:"Source Serif 4",Georgia,serif; font-size:1.02rem; line-height:1.62;
     margin:0; padding-left:.9rem; border-left:2px solid var(--accent); max-width:64ch}
.count{font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.7rem;
       color:var(--ink-3); display:block; margin-top:.45rem; padding-left:0}
.findings{margin:0; padding:.6rem .8rem .6rem 1.9rem; list-style:square;
          background:color-mix(in srgb,var(--flag) 8%,transparent);
          border-left:2px solid var(--flag); font-size:.85rem; color:var(--ink)}
.findings li::marker{color:var(--flag)}
.metrics{display:flex; flex-wrap:wrap; gap:.15rem 1.2rem;
         font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.72rem;
         color:var(--ink-2); font-variant-numeric:tabular-nums}
.metrics b{color:var(--ink-3); font-weight:400}
figure{margin:0}
figure img{display:block; width:100%; border:1px solid var(--line); border-radius:2px}
figcaption{font-size:.7rem; color:var(--ink-3); margin-top:.35rem;
           font-family:"JetBrains Mono",ui-monospace,monospace}
.score{border-top:1px solid var(--line); padding-top:.8rem; display:flex;
       flex-direction:column; gap:.1rem}
.score h3{font-size:.68rem; text-transform:uppercase; letter-spacing:.1em;
          color:var(--ink-3); margin:0 0 .45rem; font-weight:700}
.row{display:flex; align-items:center; gap:.7rem; padding:.28rem 0}
.row .label{flex:1; min-width:0}
.row .name{font-size:.85rem; display:block}
.row .hint{font-size:.72rem; color:var(--ink-3); display:block; line-height:1.35}
.dots{display:flex; gap:.28rem; flex-shrink:0}
.dot{width:1.35rem; height:1.35rem; border-radius:50%; border:1px solid var(--line);
     background:transparent; cursor:pointer; padding:0;
     font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.65rem;
     color:var(--ink-3); line-height:1; transition:background .12s,color .12s}
.dot:hover{border-color:var(--accent)}
.dot:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
.dot[aria-pressed="true"]{background:var(--accent); border-color:var(--accent); color:var(--ground)}
.muted{color:var(--ink-3); font-size:.85rem; margin:0}
@media (max-width:560px){
  .wrap{padding:1.4rem .8rem 3rem}
  .row{flex-wrap:wrap; gap:.3rem .7rem}
  .row .label{flex:1 1 100%}
  .status{margin-left:0; flex-basis:100%}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
"""

SCORE_SCRIPT = """
<script>
// Scores persist per viewer so a review survives a scroll or a reload on a phone.
// Storage can throw (private windows, blocked site data), so every access is guarded.
(function () {
  var KEY = 'aivg-scores';
  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; }
  }
  function save(state) {
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* fine */ }
  }
  var state = load();
  document.querySelectorAll('.dot').forEach(function (dot) {
    var key = dot.dataset.key;
    if (state[key] === Number(dot.dataset.value)) dot.setAttribute('aria-pressed', 'true');
    dot.addEventListener('click', function () {
      var value = Number(dot.dataset.value);
      state[key] = state[key] === value ? 0 : value;
      save(state);
      dot.parentElement.querySelectorAll('.dot').forEach(function (sibling) {
        sibling.setAttribute('aria-pressed',
          String(Number(sibling.dataset.value) === state[key]));
      });
    });
  });
})();
</script>
"""


def build_report(
    reviews: list[dict[str, Any]],
    destination: Path,
    embed: bool = True,
    standalone: bool = True,
) -> Path:
    """Write an HTML review page.

    embed=True inlines contact sheets as data URIs, so the single file is portable:
    publishable as an Artifact, openable on a phone, and able to outlive this container.
    standalone=False omits the document scaffolding, for publishing as an Artifact
    (which supplies its own charset, viewport, and wrapper).
    """
    with_sheets = max(1, sum(1 for r in reviews if r["sheet"]))
    per_sheet_budget = max(MIN_SHEET_BYTES, EMBED_BUDGET_BYTES // with_sheets)

    takes = []
    for index, review in enumerate(reviews):
        record = review["record"]
        enrichment = record.get("enrichment", {})
        request = record.get("request", {})
        render = record.get("render") or {}
        lint = review["lint"]
        metrics = review["metrics"]
        flags = (metrics or {}).get("flags", [])

        chips = [
            f'<span class="chip {"pass" if lint["ok"] else "flag"}">'
            f'prompt {"ok" if lint["ok"] else str(len(lint["violations"]))}</span>'
        ]
        if metrics:
            chips.append(
                f'<span class="chip {"pass" if not flags else "flag"}">'
                f'video {"ok" if not flags else str(len(flags))}</span>'
            )

        findings = lint["violations"] + list(flags)
        if review["error"]:
            findings.append(review["error"])
        findings_html = (
            f'<ul class="findings">'
            + "".join(f"<li>{html.escape(str(f))}</li>" for f in findings)
            + "</ul>"
        ) if findings else ""

        metrics_html = ""
        if metrics:
            metrics_html = (
                '<div class="metrics">'
                f'<span><b>frames</b> {metrics["frame_count"]}</span>'
                f'<span><b>motion</b> {metrics["mean_delta"]:.4f}</span>'
                f'<span><b>peak</b> {metrics["max_delta"]:.4f}</span>'
                f'<span><b>detail</b> {metrics["mean_detail"]:.4f}</span>'
                f'<span><b>frozen</b> {metrics["frozen_frames"]}</span>'
                "</div>"
            )

        figure = '<p class="muted">No contact sheet &mdash; this run was not rendered.</p>'
        if review["sheet"]:
            source = html.escape(review["sheet"])
            if embed:
                sheet_file = Path(review.get("sheet_path") or (destination.parent / review["sheet"]))
                if sheet_file.is_file():
                    source = _embed_sheet(sheet_file, per_sheet_budget)
            figure = (
                f'<figure><img src="{source}" alt="Contact sheet of sampled frames">'
                f'<figcaption>sampled frames, left to right</figcaption></figure>'
            )

        rows = []
        for criterion, hint in SCORECARD:
            key = f"{index}:{criterion}"
            dots = "".join(
                f'<button class="dot" type="button" aria-pressed="false" '
                f'data-key="{html.escape(key)}" data-value="{value}" '
                f'aria-label="{html.escape(criterion)}: {value} out of 5">{value}</button>'
                for value in range(1, 6)
            )
            rows.append(
                f'<div class="row"><span class="label">'
                f'<span class="name">{html.escape(criterion)}</span>'
                f'<span class="hint">{html.escape(hint)}</span></span>'
                f'<span class="dots">{dots}</span></div>'
            )

        takes.append(f"""
<article class="take">
  <div class="slate">
    <span class="id">{html.escape(review["name"])}</span>
    <span class="sep">/</span><span>{html.escape(str(request.get("style")))}</span>
    <span class="sep">/</span><span>seed {html.escape(str(request.get("seed")))}</span>
    <span class="sep">/</span><span>{html.escape(str(request.get("width")))}&times;{html.escape(str(request.get("height")))}</span>
    <span class="sep">/</span><span>{html.escape(str(request.get("num_frames")))}f</span>
    <span class="sep">/</span><span>{html.escape(str(enrichment.get("provider")))}
      &rarr; {html.escape(str(render.get("backend", "not rendered")))}</span>
    <span class="status">{"".join(chips)}</span>
  </div>
  <div class="body">
    <p class="in"><b>in</b> &nbsp;{html.escape(enrichment.get("original", ""))}</p>
    <p class="out">{html.escape(enrichment.get("enriched", ""))}
      <span class="count">{lint["word_count"]} words</span></p>
    {findings_html}
    {metrics_html}
    {figure}
    <div class="score"><h3>Score by eye</h3>{"".join(rows)}</div>
  </div>
</article>""")

    clean = sum(1 for r in reviews if r["lint"]["ok"] and not (r["metrics"] or {}).get("flags"))
    rendered = sum(1 for r in reviews if r["sheet"])

    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&"
        'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">'
    )

    content = f"""{fonts}
<title>Dailies &mdash; AI Video Gen</title>
{PAGE_STYLES}
<div class="wrap">
  <header>
    <h1>Dailies</h1>
    <p class="lede">Every take from the benchmark, with its prompt and sampled frames.
      The automated checks catch failure modes only &mdash; a take can pass all of them
      and still be wrong. Scoring is the part that needs your eyes.</p>
    <p class="tally">{len(reviews)} takes &middot; {rendered} rendered &middot;
      {clean} with no automated findings</p>
  </header>
  {"".join(takes)}
</div>
{SCORE_SCRIPT}
"""

    page = content if not standalone else (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'{content}\n</head>\n</html>\n'
    )
    destination.write_text(page, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="review", description="Review pipeline output.")
    parser.add_argument("directory", nargs="?", default=str(OUTPUTS_DIR))
    parser.add_argument("--lint-only", action="store_true",
                        help="check prompts only; skip frame extraction")
    parser.add_argument("--no-embed", action="store_true",
                        help="link contact sheets instead of inlining them (smaller file, "
                             "but no longer portable off this machine)")
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

    report = build_report(reviews, directory / "review.html", embed=not args.no_embed)
    size = report.stat().st_size
    print(f"\nReport: {report}  ({size / 1024:.0f} KB"
          f"{', self-contained' if not args.no_embed else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
