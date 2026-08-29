# Rendering on Kaggle's free GPU

The `hf` backend bills per clip. The free tier is roughly 100K credits a month, video is
priced in GPU-seconds, and the practical result is **two or three clips** before a hard
`402`. That is not a tier to economise within — it is the wrong unit of purchase for work
that involves rendering the same prompt twenty times.

Kaggle sells the opposite unit: **30 GPU-hours per week, free**, no payment method. At
roughly a minute per clip that is hundreds of renders a week, and the meter stops mattering.

`notebooks/kaggle_ltx.ipynb` is the notebook. This page is why it is shaped the way it is.

## The economics, plainly

Per-clip figures are approximate and move; the ratio between them does not.

| Path | Per ~5s clip | Ceiling |
|---|---|---|
| HF free tier (`hf` backend) | ~$0.04–0.10 | 2–3 clips, then `402` |
| HF PRO, $9/mo | same per clip | 20x the credits, same wall further out |
| **Kaggle free** | **£0** | **30 GPU-hours/week** |
| Rented 4090, ~$0.30/hr | ~$0.005 | your wallet |

Hosted APIs are priced for occasional calls from an application. Iterative prompt work is
the case they are worst at, because every iteration is a full-price render.

## The honest quality trade

Kaggle's free GPUs are 16 GB — T4 x2, or P100. That constrains which model fits:

- **LTX 2B distilled** fits comfortably. This is the reliable default, and what the
  notebook selects on a 16 GB card.
- **LTX 13B distilled** does not fit in 16 GB without offloading, and offloading is slow
  enough to spend the quota you came here to save.

So Kaggle buys **volume**, not the largest model. If what you want is 13B or 14B quality,
a rented 24 GB card is the honest answer — about $0.30/hour, still ~10x cheaper per clip
than the hosted API, and it unlocks `--backend wan2gp` as well.

Use each for what it is good at: **Kaggle to develop taste across many clips**, a rented
card for the final renders where model size actually shows.

## Why the notebook clones twice

LTX-Video's inference extras are heavy and pinned. The `ltx` backend shells out to its
CLI rather than importing it, so the two dependency sets never have to agree and an
upstream refactor cannot break the pipeline. Installing LTX into the project's own
environment would give up that isolation for nothing.

## Two settings that are not on by default

Both in the notebook's right-hand panel, and both easy to lose an hour to:

1. **Accelerator → GPU T4 x2.** Otherwise everything runs on CPU. A render that should
   take a minute takes hours, and nothing warns you.
2. **Internet → On.** Needed to clone and to pull weights. Kaggle gates this behind phone
   verification on the account — a one-time step in Settings.

Quota is billed by **session wall-clock, not GPU work**. An idle notebook costs the same
as a busy one, so stop the session when you stop working.

## What survives, and what does not

`/kaggle/working` persists with the notebook and is downloadable from the Output panel.
Everything else is gone when the session ends.

Commit the **sidecar JSON**. It is small, diffable, and records prompt, provider, model,
seed and every resolved parameter — enough to reproduce a clip after both the container
and the notebook are gone. Video files do not belong in git.

Note that `.gitignore` currently excludes `outputs/` wholesale, sidecars included, which
contradicts that advice. Narrow it to the video extensions if you want the records kept.

## Credentials

The notebook reads an optional Kaggle Secret named `GITHUB_TOKEN` (Add-ons → Secrets) for
a private clone. Do not paste a token into a cell — notebook source is saved with the
notebook, and Kaggle notebooks are shareable.

`HF_TOKEN` is not needed here at all. Nothing in this path calls Inference Providers,
which is the point.

## When Kaggle is the wrong tool

- **You need the largest models.** 16 GB decides this. Rent a card.
- **You want an unattended batch.** Kaggle sessions are interactive and time-limited;
  they stop when you close them.
- **You are iterating on prompt wording, not video.** That needs no GPU at all. Stay on
  `--enricher fixture` with `--dry-run` and `--lint-only`, which cost nothing anywhere.
