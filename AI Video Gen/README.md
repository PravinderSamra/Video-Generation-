# AI Video Gen

An automated **text-to-video pipeline** built entirely from free and open-source parts.

```
user prompt  ──▶  ENRICHMENT  ──▶  enriched prompt  ──▶  GENERATION  ──▶  video file
"a red fox"       text · CPU        (cached JSON)         diffusion · GPU     outputs/*.mp4
```

Nothing here requires a paid API key or subscription. Every component is either a local
open-source model or a service with a genuinely usable free tier.

---

## Why it is built this way

Enrichment and generation are **different problems**, so they get different resources:

- **Enrichment is a text problem.** It runs on CPU in seconds, and its output is plain
  text you can read, diff, and correct. Bounding it to whatever LLM a video app happens
  to ship would be a waste.
- **Generation is a diffusion problem.** GPU-bound, model-specific, minutes per shot.

Splitting them means you can iterate on prompt engineering fifty times a minute
(`--dry-run`) without paying a render each time. That single property is the reason for
the whole architecture — the full analysis is in [`docs/architecture.md`](docs/architecture.md).

The research behind the component choices is in
[`docs/repo_evaluation.md`](docs/repo_evaluation.md).

---

## Structure

```
AI Video Gen/
├── README.md                    ← you are here
├── requirements.txt             stdlib + PyYAML; backends install separately
├── .env.example                 copy to .env
├── docs/
│   ├── repo_evaluation.md       the 7 repos evaluated, and why
│   └── architecture.md          single vs hybrid, data flow, licence notes
├── prompts/
│   ├── cinematic_enrichment.md  the enrichment system prompt
│   ├── style_presets.yaml       8 visual styles (cinematic, noir, anime, …)
│   ├── model_dialects.yaml      per-backbone phrasing guidance
│   └── negative_prompt.txt      shared negative prompt
├── src/
│   ├── config.py                settings, request/result objects
│   ├── enrich.py                STAGE 1 — ollama | hf | passthrough
│   ├── pipeline.py              orchestration + provenance sidecar
│   ├── cli.py                   command-line entry point
│   └── backends/                STAGE 2 — ltx | wan2gp | comfyui | hf | stub
├── workflows/                   ComfyUI API-format graphs
├── tests/test_pipeline.py       smoke tests — no GPU, no network
└── outputs/                     rendered video + .json provenance record
```

---

## Quick start

### 0. Verify the pipeline works (30 seconds, no GPU, no downloads)

```bash
cd "AI Video Gen"
pip install -r requirements.txt
python -m tests.test_pipeline
python -m src.cli "a red fox in a snowstorm" --enricher passthrough --backend stub --duration 1
```

The `stub` backend renders a synthetic clip with no model at all. Its only job is to
prove the plumbing — parameter resolution, file naming, provenance — before you spend an
hour downloading weights.

### 1. See what your machine can actually run

```bash
python -m src.cli --check
```

Reports every enricher and backend as `ready` or `BLOCKED`, with the exact command to fix
each blocker.

### 2. Enrichment (local LLM, free, offline)

```bash
# install Ollama from https://ollama.com
ollama pull qwen3:8b
python -m src.cli "a red fox in a snowstorm" --dry-run
```

`--dry-run` enriches and stops. This is the loop you will live in while tuning
`prompts/cinematic_enrichment.md`.

### 3. Generation (local GPU)

```bash
git clone https://github.com/Lightricks/LTX-Video
cd LTX-Video && python -m pip install -e .[inference] && cd -
cp .env.example .env
echo "LTX_REPO=$(pwd)/../LTX-Video" >> .env

python -m src.cli "a red fox in a snowstorm" --style nature_doc --dialect ltx
```

### No GPU?

```bash
echo "HF_TOKEN=hf_..." >> .env       # free token: huggingface.co/settings/tokens
python -m src.cli "a red fox in a snowstorm" --enricher hf --backend hf
```

---

## Usage

```bash
python -m src.cli PROMPT [options]

  --enricher {ollama,hf,passthrough}         stage 1  (default: ollama)
  --backend  {ltx,wan2gp,comfyui,hf,stub}    stage 2  (default: ltx)
  --style    {cinematic,documentary,noir,anime,nature_doc,retro_8mm,cyberpunk,claymation}
  --dialect  {ltx,wan,hunyuan,generic}       phrasing tuned to the target model
  --width --height --fps --duration --seed --steps --guidance
  --dry-run                                  enrich only, skip generation
  --check                                    report what is ready
```

Pick a row based on what your machine has — every row costs £0:

| Situation | Command |
|---|---|
| GPU ≥ 12 GB, offline | `--enricher ollama --backend ltx` |
| GPU ≥ 16 GB, max quality | `--enricher ollama --backend wan2gp --dialect wan` |
| GPU 6–8 GB | `--enricher ollama --backend ltx` (2B distilled variant) |
| No GPU | `--enricher hf --backend hf` |
| CI / testing | `--enricher passthrough --backend stub` |

Every render writes `outputs/<slug>-<seed>.json` alongside the video, recording the
original prompt, the enriched prompt, the provider, the model, the seed, and every
resolved parameter — enough to reproduce it or compare backends fairly.

---

## Roadmap

**Phase 0 — Foundation ✅ complete**
Project structure; repo research; architecture decision; two-stage pipeline with three
enrichers and five backends behind swappable interfaces; CLI with `--check` and
`--dry-run`; provenance sidecars; smoke tests passing with no GPU or network.

**Phase 1 — First real video (~1 hour, mostly downloads)**
1. `ollama pull qwen3:8b`, confirm with `--check`.
2. Tune `prompts/cinematic_enrichment.md` against `--dry-run` until enriched prompts read
   like shot notes. Free and fast — do this *before* touching a GPU.
3. Clone LTX-Video, install `.[inference]`, set `LTX_REPO`.
4. First render at 512×320, 3s, `ltxv-2b-distilled` — smallest thing that proves the path.
5. Scale to 768×512, 5s, `13b-distilled` once that works.

**Phase 2 — Quality (needs Phase 1 working end to end)**
1. Install Wan2GP; verify its API route against your installed version (it moves between
   releases — see the note in `src/backends/wan2gp.py`).
2. A/B the same enriched prompt across `ltx` and `wan2gp` at a fixed seed. The sidecars
   make this a fair comparison; keep the winners as regression prompts.
3. Build a small benchmark set (8–10 prompts spanning the style presets) and re-run it
   whenever the enrichment template changes. Prompt engineering without a fixed benchmark
   is guesswork.
4. Tune `model_dialects.yaml` from what you learn — this is where most quality lives.

**Phase 3 — Multi-shot sequences**
1. Add a `shot_list` enricher: one idea → N enriched shot prompts with a consistent
   subject description carried across all of them (this is the hard part — character
   consistency between shots is the main open problem in open-source video).
2. Render each shot, then concatenate with ffmpeg.
3. Add optional post steps as ComfyUI graph nodes: frame interpolation (RIFE) for smoother
   motion, then upscaling. This is where the `comfyui` backend earns its place.

**Phase 4 — MCP surface**
1. Wrap `pipeline.run()` as an MCP server so an agent can call
   `generate_video(prompt, style, backend)` directly.
2. Attach the Hugging Face MCP server (`hf.co/mcp`) for model and Space discovery.
3. Keep the control-plane rule: tools return **paths**, never video bytes. A tool result
   has to fit in a context window; an MP4 does not.

**Deliberately not planned:** a web UI. It is a large amount of work that teaches you
nothing about video quality, and Wan2GP already ships a good one if you want to click.

---

## Licences

Mixed, and worth checking before commercial use.

| Component | Licence | Note |
|---|---|---|
| LTX-Video | Apache-2.0 / OpenRAIL-M | commercially usable |
| Wan 2.2 | Apache-2.0 | commercially usable |
| HunyuanVideo-1.5 | Tencent community licence | **read it before commercial use** |
| Ollama | MIT | — |
| ComfyUI | **GPL-3.0** | HTTP only — never vendor or import its code |

This project's own code is unencumbered; the constraint is what you connect it to.
