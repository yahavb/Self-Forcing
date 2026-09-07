# Measured results: Self-Forcing T2V on trn3-dev1

Run `20260907_045344`, commit `e6d0210`, node `ip-192-168-192-43.us-east-2`,
`neuronx-cc 2.27.2878.0+8220f7ac`, `torch-neuronx 2.12.3.0.1636+5c472775`.

Config: 16 ranks (TP=4 × SP=4), 480×640 (latent 60×80, `frame_seq_length` 1200),
sharded VAE, 21 latent frames → 81 pixel frames, probes off.

## Numbers

| prompt | frames | wall | fps | block median |
|---|---|---|---|---|
| 0 | 81 | 366.8 s | 0.22 | 42.4 s (includes kernel tracing) |
| 1 | 81 | 104.1 s | **0.78** | 13.40 s |
| 2 | 81 | 103.9 s | **0.78** | 13.41 s |

Reproducible to 0.2% between prompts 1 and 2. Quote prompt 1 or 2; prompt 0 pays to
trace one flash-attention specialisation per distinct attention-window length.

Stage split per block, steady state: **DiT 13.2 s, VAE 0.142 s, T5 0.024 s per
prompt.** The VAE is 1% of the time and the mp4/PNG writing is outside the block
timing (as it is in rolling_forcing's own reporting, so the comparison is
like-for-like).

## Why this is 0.78 and not rolling forcing's 7.8-8.6

Attention math is not the bottleneck, and the run proves it: DiT time is flat at
13.2 s while the KV window grows 7× across the video, and block 0 — the *smallest*
window — is the slowest of all.

| block | KV window | DiT |
|---|---|---|
| 0 | 3 frames | 23.59 s |
| 1 | 6 frames | 13.20 s |
| 3 | 12 frames | 13.23 s |
| 6 | 21 frames | 13.32 s |

What costs the time is the number of model calls, each carrying the dispatch cost of
the ~370 NEFFs per rank that `rolling_forcing/docs/compilation_pattern.md` records
for this stack:

- **Self-Forcing issues 5 forwards per block** — 4 denoising steps at a uniform
  timestep, then 1 at `context_noise` to rewrite clean KV. 13.26 s / 5 = 2.65 s each.
- **Rolling forcing issues 1** per phase, because its rolling window denoises `nds`
  blocks at staggered noise levels in a single fused `mode="merged"` forward. At
  8.2 fps and 12 pixel frames per block that is ~1.46 s per block.

5 × 1.46 = 7.3 s, the same order as our 13.3 s. The gap is the schedule, not the
port: one forward per block against five.

This is inherent to the chunk-wise Self-Forcing checkpoint. Amortising denoising
steps across blocks *is* what rolling forcing added.

## What could actually close it, in order of leverage

1. **Cut dispatches per forward.** 2.65 s for ~370 NEFFs is ~7 ms each, which is far
   above a NEFF launch. Worth profiling before assuming it is irreducible: fusing
   q/k/v into one matmul, or fusing norm+modulation, would cut the count directly.
   This helps rolling forcing equally, so it is a change to the shared stack.
2. **Fuse the cache pass into the next block's first denoise step** via the existing
   `mode="merged"` path, which already accepts a `cache_update_start` and an
   `nfpb_cu`. 5 forwards → 4, so ~20%.
3. **Fewer, larger ranks.** At SP=4 each rank holds 900 query tokens per forward,
   which is small enough that fixed overhead dominates. 8 ranks at the same width
   doubles per-rank work; `1200 % 8 == 0`, so it is one env change to test.
4. **Accept the schedule and quote it honestly.** 0.78 fps at 480×640 with a
   correct, streaming, 16-rank implementation is the real number for this checkpoint's
   schedule.

Not a lever: the VAE (1% of time), T5 (24 ms), or the on-disk compile caches
(steady-state execution is unaffected).
