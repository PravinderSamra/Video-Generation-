# ComfyUI workflows

Drop **API-format** workflow JSON here, then select it with the `comfyui` backend.

Export from ComfyUI with **Save (API Format)** — not the plain "Save", which produces a
UI graph the `/prompt` endpoint cannot execute.

Then replace the literal values in the JSON with these placeholders, which
`src/backends/comfyui.py` substitutes at render time:

| Placeholder | Replaced with |
|---|---|
| `%prompt%` | the enriched prompt |
| `%negative%` | the negative prompt |
| `%width%` / `%height%` | resolution |
| `%frames%` | frame count (snapped to 8n+1) |
| `%fps%` | frame rate |
| `%seed%` | seed |
| `%steps%` | sampler steps |

Starting points: [`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo)
ships example text-to-video graphs, and
[`comfyanonymous/ComfyUI_examples`](https://github.com/comfyanonymous/ComfyUI_examples)
covers the general patterns.

The default filename the backend looks for is `ltx_t2v.json`.
