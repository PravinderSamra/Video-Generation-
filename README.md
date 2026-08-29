# Video-Generation-

Local video generation projects using Remotion, MoviePy, and Playwright.

## Projects

### [AI Video Gen](./AI%20Video%20Gen/)

An automated text-to-video pipeline built entirely from free and open-source parts:
a local LLM enriches a short prompt into a cinematic shot description, then an
open-source video diffusion model renders it.

```
user prompt  ──▶  ENRICHMENT  ──▶  enriched prompt  ──▶  GENERATION  ──▶  video file
"a red fox"       text · CPU        (cached JSON)         diffusion · GPU
```

- **Enrichers:** Ollama (local, offline), Hugging Face Inference Providers, passthrough
- **Backends:** LTX-Video, Wan2GP (Wan 2.2), ComfyUI, Hugging Face, stub (no GPU)

Start here: [`AI Video Gen/README.md`](./AI%20Video%20Gen/README.md) ·
[architecture](./AI%20Video%20Gen/docs/architecture.md) ·
[repo evaluation](./AI%20Video%20Gen/docs/repo_evaluation.md)

```bash
cd "AI Video Gen"
pip install -r requirements.txt
python -m tests.test_pipeline                                    # verify, no GPU needed
python -m src.cli --check                                        # what can this machine run?
python -m src.cli "a red fox" --enricher passthrough --backend stub --duration 1
```
