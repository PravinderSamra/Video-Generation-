# Running this from a phone

You do not run anything on the phone. The phone is a terminal; the work happens in a
cloud container driven from the Claude Code session. Nothing is installed locally, and
neither stage of the pipeline touches the handset.

```
   your phone            this container              free hosted GPU
  (Claude Code)   ──▶   python, network,      ──▶   HF Inference Providers
   type, review          the pipeline                or a ZeroGPU Space
        ▲                     │
        └──── artifact ───────┘   review page you open in the browser
```

## The three real constraints

Worth internalising before you start, because they shape everything else.

**1. The container is ephemeral.** It is reclaimed after inactivity. Anything in
`outputs/` is gone. Whatever matters has to be committed and pushed, sent to you as a
file, or published as an artifact — *during* the session.

**2. Free hosted video is metered in clips, not hours.** The Hugging Face free tier is
100K Inference Provider credits per month, which is on the order of **under ten video
clips**. That is enough to prove the path end to end. It is not an iteration loop.

**3. Do not paste secrets into the chat.** Anything typed into the conversation is in the
transcript. Set `HF_TOKEN` as an **environment variable on the environment itself**,
configured in the Claude Code web settings — see
[the docs](https://code.claude.com/docs/en/claude-code-on-the-web). The session picks it
up from `os.environ` with no token ever appearing in a message.

## Setup, entirely from the phone

1. **Free Hugging Face token** — in the phone browser, huggingface.co →
   Settings → Access Tokens → New token, read scope. Copy it.
2. **Set it on the environment** — Claude Code web settings → your environment →
   environment variables → `HF_TOKEN`. Not in the chat.
3. **Confirm** — ask the session to run `python -m src.cli --check`. The `hf` enricher
   and backend should flip from BLOCKED to ready.

## The loop that actually works on a phone

Spend your free credits on *video*, never on text. Enrichment through `hf` is cheap and
text-only, so iterate there freely; every video render is a meaningful fraction of your
monthly allowance.

```bash
# Free and unmetered in practice — iterate here as much as you like
python -m src.benchmark --dry-run --enricher hf
python -m src.review outputs/benchmark --lint-only

# Metered. One clip. Only once the prompt above is one you would pay for.
python -m src.cli "…" --enricher hf --backend hf --width 512 --height 320 --duration 3
```

## Reviewing on a phone

This is the part that needs the tooling, because there is no file manager and no video
player pointed at the container's disk.

- **The review page.** `python -m src.review` writes a self-contained HTML file —
  contact sheets are inlined as data URIs, so it survives being moved, published, or
  outliving the container. Ask the session to publish it as an artifact and you get a URL
  that opens in the phone browser, works in light and dark, and lets you **tap a score
  1–5 per criterion**, persisted in the browser so a scroll or reload does not lose it.
- **The video itself.** Ask the session to send you the MP4; it renders inline in the app.
  Watch it *after* the contact sheet — the sheet exposes drift and collapse that are easy
  to miss at 24fps, and the video exposes motion problems the sheet cannot show.
- **Keeping results.** Commit the sidecar JSON. It is small, diffable, and records the
  prompt, provider, model, seed and every parameter, so a run stays reproducible after
  the container is gone. Video files do not belong in git.

## When the free tier runs out

It will, quickly. In rough order of what to reach for:

- **ZeroGPU Spaces** — free A100 time on accounts with no payment method, with a daily
  quota, callable with your token. Volatile: Spaces appear, change and disappear, so
  treat any specific one as temporary.
- **Hugging Face PRO ($9/mo)** — 2M credits and a ZeroGPU quota. The cheapest way to make
  this a real loop rather than a demo.
- **A rented GPU by the hour** — once you are rendering seriously, an hour of a rented
  card beats a month of hosted credits, and it unlocks the `ltx` and `wan2gp` backends
  that this project is actually built around.

The mobile path is genuinely good for **building and reviewing** this pipeline. It is a
poor fit for the part where you generate a hundred clips and develop taste. Plan to move
the generation half onto real hardware; the architecture already assumes you will, which
is why the backends are swappable and the enriched prompt is a portable artefact.
