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

## Control: rolling forcing on the SAME node, same geometry

Run `rf-job-fs1200_20260907_201651`, `rf-job.yaml` unmodified except memory
250Gi -> 500Gi (it OOMKilled at 256Gi: 16 ranks x 11.4GB T5). Same node, same 16
ranks, same latent_w 80 / fs1200, same sharded VAE:

```
DiT-only : median 16.13 fps | max 17.01   (96 blocks)
DiT+VAE  : median 13.05 fps | max 13.69
per block: DiT 705-751 ms, VAE ~175 ms
```

So rolling forcing does **13.05 fps here**, better than the 7.8-8.6 in its own docs,
and there is no trn2-vs-trn3 excuse. The real gap is 17.7x, not 10x, and it
decomposes cleanly:

| | RF | SF | ratio |
|---|---|---|---|
| DiT per block | 750 ms | 13260 ms | 17.7x |
| passes per block | 1 x 30 layers | 5 x 30 layers | 5x |
| per layer-pass | 25.0 ms | 88.4 ms | 3.5x |

**5x is structural** — self-forcing's 4 denoising steps are sequentially dependent
and cannot be fused. **3.5x is mine** and is fixable. Even fixing all of it caps
self-forcing at 5 x 25 ms x 30 = 3.8 s/block = **3.2 fps**, against rolling
forcing's 13. That is the ceiling for this checkpoint's schedule.

## Why the 3.5x per-pass penalty is mine

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

## Profile: the cost is host-side dispatch churn, not math

Run `bb62233` with `SF_PROFILE=1`, profiling one warm prompt (35 passes = 7 blocks x 5
forwards, 1050 layer-executions). The device trace was dropped (`NEFF/NTFF mismatch`,
15565 of 48537 executions) so there are no per-op timings, but the CPU-side event
counts are decisive:

| event | total | per pass | per layer |
|---|---|---|---|
| `dmem_buf_copyin` | 167435 | 4784 | **159** |
| `neuron::alloc::lazy` | 110080 | 3145 | 105 |
| `aten::contiguous` | 27214 | 778 | 26 |
| `event_signal` | 9064 | 259 | 9 |
| `kbl_exec_post` | 7936 | 227 | 8 |

4784 host-to-device copy-ins per pass at ~0.5 ms each is ~2.4 s, essentially the whole
2.8 s of per-pass cost that attention, DMA volume and the schedule could not explain.
159 copy-ins and 105 lazy allocations for a single transformer layer is churn, not
work.

This also corrects an earlier claim in this file: 5-passes-vs-1 is **not** the binding
constraint. Self-forcing pushes 18000 query tokens per block against rolling forcing's
21600 -- less work -- so at equal per-frame efficiency this schedule would run at ~19
fps. The gap is entirely per-pass overhead.

### The open question

Does rolling forcing show the same per-layer churn? Its job carries the same profiler
wiring, so this is answerable with their code and no changes of mine:

- if RF is also ~4784 copy-ins per pass, the overhead is inherent to the stack and the
  only lever is passes per block, which is the schedule
- if RF is ~200 per pass, something in this port's path (`mode="denoise"` with
  `updating_cache=True`, versus RF's fused `merged`) is generating the churn, and it is
  fixable
