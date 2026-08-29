# Architecture — Free MCP & Resource Design

**Decision: hybrid.** Two specialised free resources, not one general one.

---

## 1. The question: single resource or hybrid mix?

### Option A — Single resource

Pick one system that does everything. The only credible free candidate is **Wan2GP**,
which genuinely bundles a prompt enhancer with the generation stack.

| | |
|---|---|
| ✅ | One install, one process, one dependency tree. Fastest path to a first video. |
| ✅ | Its enhancer is *model-aware* — it knows Wan's phrasing differs from LTX's. |
| ❌ | Enrichment quality is bounded by whatever LLM that app ships. We cannot swap in a better one. |
| ❌ | A GPU is mandatory **for the whole pipeline**, including the text step that does not need one. |
| ❌ | Couples our roadmap to one fast-moving upstream project. A breaking release breaks everything. |
| ❌ | No way to iterate on enrichment without paying full video-render cost each time. |

That last point is the killer. Prompt engineering is an **iteration loop** — you want to
run it fifty times in a minute. Behind a single monolith, every iteration costs a render.

### Option B — Hybrid mix ✅ **chosen**

Split at the natural seam the research already exposed: enrichment is a *text* problem,
generation is a *diffusion* problem.

| | |
|---|---|
| ✅ | Enrichment iterates in **seconds on CPU**, decoupled from render cost. |
| ✅ | Each stage swaps independently — new video model, same enricher, and vice versa. |
| ✅ | **Degrades gracefully:** no GPU → hosted HF backend; no network → local Ollama + local LTX. |
| ✅ | The enriched prompt is a **plain-text artefact** we can cache, diff, version, and review. |
| ⚠️ | Two services to run. Mitigated by making every stage independently swappable *and* stubbable. |

**Verdict: hybrid.** The cost of Option B is one extra process. The cost of Option A is
losing the ability to improve the half of the pipeline that is cheapest to improve.

---

## 2. Data flow

```
                          ┌─────────────────────────────────────────────┐
   User prompt            │  STAGE 1 — ENRICHMENT   (text · CPU-viable) │
   "a fox in the snow"    │                                             │
          │               │  Provider (pick one, all free):             │
          └──────────────▶│    • ollama    → localhost:11434  [default] │
                          │    • hf        → router.huggingface.co/v1   │
                          │    • passthrough → no-op, for A/B baseline  │
                          │                                             │
                          │  Inputs: prompts/cinematic_enrichment.md    │
                          │          prompts/style_presets.yaml         │
                          │          target-model dialect hint          │
                          └──────────────────────┬──────────────────────┘
                                                 │
                                                 ▼
                                    ╔════════════════════════╗
                                    ║   ENRICHED PROMPT      ║  ← cached artefact
                                    ║   + negative prompt    ║    (JSON sidecar,
                                    ║   + resolved params    ║     diffable, reusable)
                                    ╚════════════┬═══════════╝
                                                 │
                          ┌──────────────────────▼──────────────────────┐
                          │  STAGE 2 — GENERATION   (diffusion · GPU)   │
                          │                                             │
                          │  Backend (pick one, all free):              │
                          │    • ltx      → local LTX-Video CLI [dflt]  │
                          │    • wan2gp   → Wan2GP HTTP API             │
                          │    • comfyui  → ComfyUI /prompt job API     │
                          │    • hf       → HF Inference Provider       │
                          │    • stub     → synthetic MP4, no GPU       │
                          └──────────────────────┬──────────────────────┘
                                                 │
                                                 ▼
                                    outputs/<slug>-<seed>.mp4
                                    outputs/<slug>-<seed>.json   ← full provenance
```

### Why the enriched prompt is a first-class artefact

It is written to disk as JSON *before* any GPU work starts. That single decision buys us:

- **Cheap iteration** — tune templates without rendering (`--dry-run`).
- **Reproducibility** — re-render the exact same prompt on a different backend to compare.
- **Reviewability** — the enrichment is text a human can read and correct.
- **Provenance** — every output MP4 has a sidecar recording the original prompt, the
  enriched prompt, the provider, the model, the seed, and the resolved parameters.

---

## 3. Resource selection matrix

Choose a row based on what the machine actually has. All rows cost £0.

| Situation | Enricher | Backend | Notes |
|---|---|---|---|
| GPU ≥ 12 GB, offline | `ollama` | `ltx` | Fully local, fully free, no network. **Recommended.** |
| GPU ≥ 16 GB, want max quality | `ollama` | `wan2gp` (Wan 2.2 14B FP8) | Slower; best output. |
| GPU 6–8 GB | `ollama` | `ltx` (`ltxv-2b-distilled`) | Or Wan2GP with a memory profile. |
| No GPU | `hf` | `hf` | Uses free monthly Inference Provider credits. |
| CI / testing / no deps | `passthrough` | `stub` | Deterministic, runs anywhere. |

---

## 4. MCP integration

Two MCP servers, both free, each doing what MCP is actually good at — *discovery and
control*, not bulk data movement:

1. **Hugging Face MCP** (`hf.co/mcp`, official, free account) — model and Space discovery,
   and calling community Gradio Spaces as tools. This is how we reach a hosted video model
   with no local GPU.
2. **Wan2GP's MCP server** (ships with the app) — drives local generation as a tool call.

**Design rule:** MCP is the *control plane*. Video bytes never travel through an MCP tool
response — the backend writes an MP4 to disk and returns a **path**. Tool results are
context-window-sized; video files are not.

---

## 5. Interface boundaries (why this stays maintainable)

Every upstream project in this design is fast-moving. We survive that by touching each one
only at a stable seam:

| Upstream | We depend on | We never depend on |
|---|---|---|
| LTX-Video | its documented `inference.py` CLI flags | its Python internals |
| Wan2GP | its HTTP API / `--process` batch mode | its Gradio UI or modules |
| ComfyUI | its REST `/prompt` endpoint | its source (**GPL-3.0** — keep it a separate process) |
| Ollama / HF | the OpenAI-compatible chat-completions shape | provider-specific extensions |

Two abstract classes enforce this: `Enricher` and `VideoBackend` (`src/backends/base.py`).
Adding a backend means implementing one method — nothing else in the codebase changes.

### A licence note worth remembering

ComfyUI is **GPL-3.0**. Calling it over HTTP from a separate process is fine. Vendoring or
importing its code would impose GPL on this project. LTX-Video (Apache-2.0/OpenRAIL-M) and
Wan 2.2 (Apache-2.0) carry no such constraint; HunyuanVideo uses Tencent's community
licence — **read it before any commercial use.**
