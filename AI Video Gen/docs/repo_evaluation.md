# Repository Evaluation — Free / Open-Source Building Blocks

**Researched:** 2026-08-29 (live GitHub + web search, not from memory)
**Question:** which free, publicly available repos can we compose into a
`prompt -> enriched prompt -> video file` pipeline?

Every option below is either **runnable locally at zero cost** or has a **usable free
API tier**. Nothing here requires a paid subscription to get a first video out.

---

## Summary table

| # | Repo | Role in our pipeline | Stars | License | Cost to run |
|---|------|---------------------|-------|---------|-------------|
| 1 | [`deepbeepmeep/Wan2GP`](https://github.com/deepbeepmeep/Wan2GP) | **Video generation + prompt enrichment** (one box) | ~9.1k | Free local use (see repo LICENSE.txt) | Free, local GPU |
| 2 | [`Lightricks/LTX-Video`](https://github.com/Lightricks/LTX-Video) | **Video generation** — fastest on low VRAM | ~10.9k | Apache-2.0 / OpenRAIL-M | Free, local GPU |
| 3 | [`Wan-Video/Wan2.2`](https://github.com/Wan-Video/Wan2.2) | **Video generation** — best open quality ceiling | ~17.3k | Apache-2.0 | Free, local GPU |
| 4 | [`Tencent-Hunyuan/HunyuanVideo-1.5`](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | **Video generation** — lightweight 8.3B, diffusers-native | ~4.5k | Tencent Hunyuan community licence | Free, local GPU |
| 5 | [`ollama/ollama`](https://github.com/ollama/ollama) | **Prompt enrichment** — local LLM, no API key | ~180k | MIT | Free, local CPU/GPU |
| 6 | [`huggingface/hf-mcp-server`](https://huggingface.co/mcp) | **MCP transport** — model/Space discovery + hosted inference | official | Apache-2.0 | Free tier (monthly credits) |
| 7 | [`comfyanonymous/ComfyUI`](https://github.com/comfyanonymous/ComfyUI) + [`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo) | **Execution graph / HTTP job API** | very large | GPL-3.0 | Free, local GPU |

---

## 1. Wan2GP (`deepbeepmeep/Wan2GP`) — *"AI video for the GPU-poor"*

**Why it matters to us:** it is the only repo found that covers **both halves of our
pipeline in a single install**. It ships a *per-model prompt enhancer* alongside the
generation stack, and it exposes a headless batch mode plus an API/MCP server — which
means we can drive it as a service rather than screen-scraping a Gradio UI.

- **Supports:** Wan 2.1 / 2.2, LTX-2, LTX-Video, Hunyuan Video, Qwen Image, Flux, plus TTS.
- **VRAM:** select models run in **as little as 6 GB**; 16 GB recommended for speed.
- **Interfaces:** one-click `.bat`/`.sh` installers, Docker, `python wgp.py --process`
  (headless batch), and an API with MCP server support.
- **Activity:** actively maintained (v12.x, updated August 2026).

**Use for prompt enrichment:** its enhancer rewrites a short prompt into the *specific
phrasing each backbone was trained on* — Wan and LTX want different things, and this
encodes that knowledge for free.

**Use for video generation:** memory-profile system (`profile 3/3+`) plus quantisation is
what makes 14B-class models viable on a consumer card at all.

**Caveat:** it is a large, fast-moving application, not a library. We integrate over its
**API/CLI boundary**, never by importing its internals — that keeps our code stable across
its weekly releases.

---

## 2. LTX-Video (`Lightricks/LTX-Video`) — *speed floor*

**Why it matters to us:** the cheapest path to a *first* video. Distilled variants make
iteration loops seconds-long instead of minutes-long, which is exactly what you want while
you are tuning prompt-enrichment templates.

- **Variants:** `ltxv-13b-0.9.8-dev` (quality), `ltxv-13b-0.9.8-distilled` (fast),
  `ltxv-2b-0.9.8-distilled` (smallest), **FP8 quantised builds for all three**.
- **Runs from:** `python -m pip install -e .[inference]`, then `python inference.py
  --prompt "..." --height H --width W --num_frames N --seed S
  --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml`
- **Built-in enrichment:** `LTXVideoPipeline(..., enhance_prompt=True)` — a *second*
  free enrichment path, in-process, no extra service.
- **Licence:** Apache-2.0 + OpenRAIL-M, commercially usable.
- **ComfyUI:** first-party nodes at `Lightricks/ComfyUI-LTXVideo`.

**Verdict:** our **default backend**. Best quality-per-VRAM-per-second of anything tested,
and the only one with a clean, documented single-command CLI.

---

## 3. Wan 2.2 (`Wan-Video/Wan2.2`) — *quality ceiling*

**Why it matters to us:** when a shot needs to be *good* rather than *fast*, this is the
open weight set to reach for. Apache-2.0, so there is no licensing friction later.

- **VRAM by variant:** 1.3B ≈ 4–6 GB (GGUF), 5B TI2V ≈ 8–12 GB, 14B ≈ 6–24 GB (FP8).
  Unquantised 14B at 720p is a datacentre-class ask (65–80 GB) — **use the quantised
  builds**, which is precisely what Wan2GP automates.
- **Prompt style:** rewards long, dense, natural-language descriptions. Our enrichment
  templates are tuned for this.

**Verdict:** our **quality backend**, driven through Wan2GP so we inherit its
offloading/quantisation work instead of reimplementing it.

---

## 4. HunyuanVideo-1.5 (`Tencent-Hunyuan/HunyuanVideo-1.5`) — *the diffusers-native option*

**Why it matters to us:** 8.3B params with a 3D causal VAE (16× spatial, 4× temporal
compression) and — critically — **first-party Hugging Face `diffusers` integration**. That
makes it the easiest backend to call from plain Python without adopting anyone's
application framework.

- **VRAM:** ~14 GB minimum with model offloading enabled.
- **Output:** 480p / 720p, up to 121 frames, configurable aspect ratio.
- **Run:** `generate.py` via `torchrun`; ComfyUI plugins and LightX2V also supported.

**Verdict:** our **portable backend** — the one that works on a rented/borrowed GPU with
nothing but `pip install diffusers`.

---

## 5. Ollama (`ollama/ollama`) — *prompt enrichment, zero API key*

**Why it matters to us:** enrichment is a text task, and a 7–8B local model does it well.
Ollama gives us an OpenAI-compatible endpoint on `localhost:11434` with a one-line install
and no key, no quota, no network. It is the single highest-leverage free component here:
it makes the *entire left half* of the pipeline cost nothing and work offline.

- 180k stars, MIT, first-party Python/JS clients.
- Ships current instruction models (Qwen, Gemma, gpt-oss, DeepSeek, GLM, …).

**Verdict:** our **default enricher**.

---

## 6. Hugging Face MCP server (`hf.co/mcp`) — *the MCP layer*

**Why it matters to us:** the official HF MCP server (source: `huggingface/hf-mcp-server`)
speaks STDIO and Streamable HTTP, ships seven built-in tools, and can **expose community
Gradio Spaces as callable tools**. That last point is the interesting one: a public
text-to-video Space becomes an MCP tool we can call with **no local GPU at all**.

- A **free** HF account is enough; every user gets monthly Inference Provider credits.
- `router.huggingface.co/v1` fronts 45k+ models across 18+ providers behind one
  OpenAI-compatible endpoint — so the *same* client code covers both enrichment and any
  hosted video model.

**Verdict:** our **no-GPU fallback** for both stages, and the MCP integration point.

---

## 7. ComfyUI (+ `ComfyUI-LTXVideo`) — *optional execution graph*

**Why it matters to us:** ComfyUI has an HTTP `/prompt` job API and a persistent model
cache. If we later want multi-step graphs (generate → interpolate → upscale → stitch),
this is the free orchestration layer, and LTX ships first-party nodes for it.

**Caveat:** **GPL-3.0**. Fine as a *separate process we talk to over HTTP*; do not vendor
its code into ours. Our adapter therefore speaks only to its REST API.

---

## Also reviewed, and why they are not in the core path

- **[`Anil-matcha/Open-Generative-AI`](https://github.com/Anil-matcha/Open-Generative-AI)** (27.3k ⭐) — MIT studio UI wrapping 500+ models. Despite the framing, the interesting models behind it are **paid third-party APIs** (Kling, Sora, Veo) via an aggregator. Good reference for UI patterns; fails the "entirely free" constraint.
- **[`HBAI-Ltd/Toonflow-app`](https://github.com/HBAI-Ltd/Toonflow-app)** (14.8k ⭐) — script → storyboard → animated short. Excellent **architectural reference** for the enrichment→shot-list→render decomposition we are building; it is a full Electron product, so we borrow the *pattern*, not the code.
- **[`PKU-YuanGroup/Helios`](https://github.com/PKU-YuanGroup/Helios)** (2.1k ⭐) — real-time long-video generation. Genuinely exciting and worth revisiting, but it is young (created March 2026) and aimed at interactive/world-model use. **Watch, don't depend.**
- **[`SamurAIGPT/Generative-Media-Skills`](https://github.com/SamurAIGPT/Generative-Media-Skills)** (4.2k ⭐) — agent skills for media generation, but routes through `muapi.ai`, a **paid** aggregator.

---

## What this means for the build

The research converges on a clear split, and it is the reason the architecture in
[`architecture.md`](./architecture.md) is **hybrid**:

- **Enrichment** is a *text* problem — cheap, fast, CPU-viable, and best served by a
  general instruction LLM (Ollama locally, HF router remotely).
- **Generation** is a *diffusion* problem — GPU-bound, model-specific, and best served by
  a purpose-built video stack (LTX-Video / Wan / Hunyuan).

No single free resource does both *well*. Wan2GP comes closest, and that is exactly why it
is our recommended all-in-one starting point — but we still wrap both stages behind
interfaces so either half can be swapped without touching the other.
