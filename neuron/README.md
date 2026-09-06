# Self-Forcing on Trainium

Self-Forcing T2V inference on AWS Trainium (Neuron): T5 → causal DiT with KV cache →
streaming VAE → mp4, across 8 NeuronCores at LNC=1 (TP=4 × SP=2).

## How this relates to rolling_forcing

Rolling Forcing and Self-Forcing share the backbone (Wan2.1-T2V-1.3B), the causal DiT, the
NKI kernels, T5 and the VAE. They differ **only in the denoising schedule**. So this
directory holds the schedule and the plumbing, and the Neuron stack is a pinned runtime
dependency of [`yahavb/rolling_forcing`](https://github.com/yahavb/rolling_forcing) rather
than a copy:

| | Self-Forcing (here) | Rolling Forcing (upstream stack) |
|---|---|---|
| loop | one block at a time | rolling window of `nds` blocks |
| per block | `nds` forwards at a uniform timestep, then 1 at `context_noise` to rewrite clean KV | 1 fused `mode="merged"` forward per phase, blocks at staggered timesteps |
| input shape | always `[1, 3, 16, 60, 104]` | padded to `nds * nfpb` frames |
| KV window | global — every frame so far | rolling window + attention sink |
| max length | 21 latent frames (81 pixel frames) | long / unbounded |

Nothing in `models/`, `kernels/` or `utils/` needed to change. The one thing to know is
that every forward passes **`updating_cache=True`**, which selects the `_assemble_kv`
branch that builds the attention window from the whole cache with the anchor re-RoPEd at
absolute frame 0. The `updating_cache=False` branch is the rolling-window one (capped
carry, *relative* anchor RoPE) and is wrong for these checkpoints, which were distilled
with `local_attn_size=-1`.

Pinned stack commit: `0cfc734656aa0ff72438f4ba2771056f4df52807`. `models/`, `kernels/`,
`utils/` and `e2e_pipeline.py` are byte-identical at that commit and at
`origin/rf-distill-1.3b`, so either is safe; the pin is what the port was read against.

## Layout

```
neuron/
  schedule.py                    # the schedule as plain data (no torch)
  tests/test_schedule.py         # laptop gate: 35-call sequence vs the reference loop
  pipeline_self_forcing.py       # SelfForcingInferencePipeline + build_sf_pipeline
  sf_inference.py                # torchrun entry point
  configs/self_forcing_dmd.yaml  # 4-step schedule, frame_seq_length 1560

self-forcing-job.yaml            # k8s Job, at the repo root next to the other
                                 # job specs (rolling_forcing convention)
```

## Running

On the cluster:

```bash
kubectl apply -f self-forcing-job.yaml
kubectl logs -f job/self-forcing
```

On a Trainium host directly, from the repo root, with `wan_models/Wan2.1-T2V-1.3B/` and
`checkpoints/self_forcing_dmd.pt` in place:

```bash
export PYTHONPATH=/path/to/rolling_forcing
torchrun --nproc_per_node 8 neuron/sf_inference.py \
  --prompt_file prompts/MovieGenVideoBench_extended.txt \
  --config_path neuron/configs/self_forcing_dmd.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --output_folder videos_sf \
  --num_output_frames 21 --tp_degree 4 --fps 16 --max_prompts 1 --use_ema
```

Laptop gate, no Neuron needed:

```bash
python3 neuron/tests/test_schedule.py
```

## Constraints worth knowing before changing anything

- **World size is 8.** `frame_seq_length` 1560 (= 30×52, from the 60×104 latent behind
  480×832) divides 8 but not 16, and an SP shard may not split a frame. 16 ranks would
  need a smaller latent (480×640 → 1200), as rolling_forcing's `tp4sp4` config does.
- **`num_frame_per_block` is 3.** The Neuron attention hardcodes
  `block_length = 3 * frame_length`.
- **21 latent frames is the ceiling**, from `max_attention_size = 21 * frame_length`.
  This is a 21-frame model; long rollouts are what rolling forcing is for.
- **First run compiles for a while.** The attention window grows per block
  (3, 6, …, 21 frames → 7 distinct `actual_seqlen_k` values into the flash kernel), so
  expect ~7 NKI specialisations on top of the compiled sub-modules. The K/V buffer width
  is padded and constant, so this is compile time, not extra shapes.
- **Re-noising draws on the host.** Each of the 3 re-noise steps per block generates
  `torch.randn` on CPU and copies to device, matching how the initial noise is produced.
  rolling_forcing's `utils/noise_producer.py` (async) and `utils/rng.py`
  (`--rng_state_path`, reproducible draws) are the hooks if this shows up in the profile
  or you need bit-reproducibility.
