# Testing & Review

"Test this" means three different things here, and they need different instruments.
Conflating them is how you end up with a green test suite and unwatchable video.

| Layer | What it judges | Automatable? | Tool |
|---|---|---|---|
| 1. Pipeline | Does the code work? | Fully | `python -m tests` |
| 2. Prompt quality | Is the enriched prompt well-formed? | Mostly | `python -m src.review --lint-only` |
| 3. Video quality | Does it look good? | **No** | contact sheets + scorecard |

Layer 3 is the one that matters, and it is the one no tool can answer. Everything below
exists to make the human review fast and fair, not to replace it.

---

## Layer 1 — Pipeline correctness

Deterministic, no GPU, no network, no model weights. Runs anywhere in about a second.

```bash
python -m tests              # both suites
python -m tests.test_pipeline
python -m tests.test_review
```

Covers frame-count snapping to the 8n+1 the VAEs require, template substitution, backend
registry construction, output filename collision handling, the dry-run path, the PNG
codec against all five scanline filter types, and the failure detectors below.

**This layer proves the plumbing, nothing more.** It cannot fail because a video looks
bad, and it should not try to.

## Layer 2 — Prompt quality

The enrichment template states explicit rules. `lint_prompt()` checks the output against
them, so the enricher cannot quietly drift after a template edit or a model swap:

- 60–110 words — shorter underspecifies, longer dilutes
- one paragraph, no line breaks
- **no negations** — diffusion models do not process negation; exclusions belong in the
  negative prompt
- no operator instructions ("please", "make sure", "we see")
- no multi-shot markers ("meanwhile", "cut to", sentence-initial "Then")
- no leftover markdown or quote wrappers
- **the user's own content words survive** — enrichment must build on their idea, not
  replace it. If they said "eagle", the output has an eagle.

```bash
python -m src.cli "a red fox in a snowstorm" --dry-run    # enrich, no render
python -m src.review --lint-only                          # lint everything in outputs/
```

This is the loop to live in while tuning `prompts/cinematic_enrichment.md`. It costs
nothing and runs in seconds, which is the whole reason the architecture splits the two
stages.

Note the "then" rule is deliberately narrow. Mid-sentence, "then" usually joins one
continuous action ("coils and then dives") and is fine; only sentence-initial "Then" or
"then," reads as a cut. A linter that cries wolf gets ignored.

## Layer 3 — Video review

### Failure detectors, not quality scores

`frame_metrics()` catches the specific ways video diffusion fails:

| Flag | Failure mode |
|---|---|
| `FROZEN` | almost no motion — a still image in a video container |
| `FLICKER` | frame-to-frame change is extreme — the clip is thrashing |
| `SLIDESHOW` | motion is stepped rather than continuous |
| `FLAT` | frames carry almost no spatial detail |
| `COLLAPSE` | detail drains away toward the end of the clip |

Every one of these has a test that synthesises the exact failure and asserts the flag
fires. A detector that never fires is worse than no detector, because it manufactures
confidence — one of these (`FLICKER`) was in fact set so high it could never trigger, and
the test is what caught it.

> **The thresholds are provisional.** They were calibrated against synthetic fixtures,
> not real renders, because this project has never had a GPU. Recalibrate them in Phase 1:
> run the benchmark set, record metrics for clips you judge good and bad by eye, and move
> the constants in `src/review.py` until the flags agree with your judgement. Until then a
> flag means "look at this one", never "this is bad".

**What these cannot tell you:** whether the fox looks like a fox, whether the lighting is
beautiful, or whether the shot matches your intent. That needs eyes.

### The benchmark set

Prompt engineering without a fixed benchmark is guesswork — you remember the good results
and forget the bad ones. `prompts/benchmark.yaml` holds ten prompts at fixed seeds, chosen
to span the failure modes that actually bite:

fur and fast motion · water and reflections · **hands** (the classic) · wet reflective
surfaces · crowds · near-static scenes that must not freeze · fast lateral tracking ·
fine repeating texture (the flicker trigger) · mechanical articulation · full-body human motion

```bash
python -m src.benchmark --dry-run                 # enrichment only — free, seconds
python -m src.benchmark --backend ltx             # full render
python -m src.benchmark --only fox_snow,potter_hands
```

Re-run it at the **same seeds** whenever you change the template, switch backend, or pull
new weights. Fixed seeds are what make two runs comparable.

### The golden enrichment set

`prompts/golden_enrichments.yaml` holds one reference enrichment per benchmark prompt,
served by `--enricher fixture`.

> **Provenance matters here.** These were hand-authored against
> `prompts/cinematic_enrichment.md`. They are a written-to-spec *target*, not a recording
> of any model's output. Treating them as "what the enricher produces" would be wrong.

Two uses:

1. **Deterministic enrichment offline** — demos and CI, with no Ollama and no API key.
2. **A baseline for judging a real enricher.** Run the same benchmark through
   `--enricher ollama` and compare against these. Output that is shorter, vaguer, or
   lint-failing means the template or the model needs work — and now you have something
   concrete to compare against instead of a vague sense that it "reads worse".

A test asserts the golden set stays lint-clean, so a linter rule change or a careless
fixture edit surfaces immediately rather than quietly degrading the baseline.

### Human review

```bash
python -m src.review outputs/benchmark
```

Produces `review.html`: per run, a **contact sheet** (frames tiled into one grid, so a
whole clip is legible at a glance), the original and enriched prompt, any automated
findings, and a blank scorecard:

| Criterion | Question |
|---|---|
| Prompt adherence | Is everything the prompt asked for actually present? |
| Subject coherence | Does the subject hold its shape and identity across all frames? |
| Motion quality | Is the movement fluid and plausible, not stepped or floaty? |
| Temporal stability | Any flicker, morphing, or background churn? |
| Cinematography | Did the requested shot size, camera move, and lighting land? |
| Style match | Does it read as the chosen style preset? |

Score each out of 5. The value is not the numbers — it is that scoring the same six things
every time stops you from judging clip 8 by a different standard than clip 1.

**Watch the contact sheet, then watch the video.** The sheet exposes drift, collapse, and
identity changes that are easy to miss at 24fps; the video exposes motion problems the
sheet cannot show. You need both.

---

## Recommended order

1. `python -m tests` — plumbing is sound.
2. `python -m src.benchmark --dry-run` then `python -m src.review --lint-only` — enrichment
   is well-formed. **Iterate here until the prompts read like real shot notes.** Free.
3. One render at the smallest useful size (512×320, 3s, `ltxv-2b-distilled`) — the path works.
4. Full benchmark at production settings, then `python -m src.review` and score it.
5. Recalibrate the metric thresholds against what you just scored by eye.
6. Change one thing. Re-run. Compare. Never change two things at once — with a stochastic
   generator you will not be able to attribute the difference.
