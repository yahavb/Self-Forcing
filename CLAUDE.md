# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Self Forcing (arXiv 2506.08009): autoregressive video diffusion distilled from Wan2.1-T2V. The core idea is that training *simulates inference* — the generator does an autoregressive KV-cached rollout during training, so there is no train/test distribution mismatch. Built on top of the Wan2.1 codebase (vendored in `wan/`) and CausVid.

Requires an NVIDIA GPU (≥24 GB) and Linux; nothing here runs on the Mac host — all commands below assume a CUDA machine.

## Setup and commands

```bash
conda create -n self_forcing python=3.10 -y && conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop          # installs as editable package `self_forcing`

# Checkpoints (paths below are hardcoded in the code, not configurable)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

CLI inference (single or multi-GPU via `torchrun`):
```bash
python inference.py --config_path configs/self_forcing_dmd.yaml \
  --output_folder videos/self_forcing_dmd \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt --use_ema
```

Web demo (Flask + SocketIO, streams frames as they decode; `--trt` for TensorRT VAE):
```bash
python demo.py            # http://0.0.0.0:5001
```

Distillation training (the paper's recipe: 600 iters, ~2h on 64×H100):
```bash
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR \
  train.py --config_path configs/self_forcing_dmd.yaml --logdir logs/self_forcing_dmd --disable-wandb
```

There is **no test suite and no lint config**. Verification means running inference and eyeballing the videos, or a short training run with `--no_save --disable-wandb`.

Always run from the repo root: `train.py`, `inference.py`, and `demo.py` all load `configs/default_config.yaml` by relative path, and `utils/wan_wrapper.py` loads T5/tokenizer/VAE weights from the relative `wan_models/Wan2.1-T2V-1.3B/` path.

## Architecture

Four layers, each dispatching into the next by config value:

1. **`train.py`** → `config.trainer` selects `trainer/{diffusion,gan,ode,distillation}.py`. Trainers own FSDP wrapping (`utils/distributed.fsdp_wrap`), optimizers, EMA (`EMA_FSDP`), wandb, and checkpoint save/load.
2. **`trainer/distillation.py`** → `config.distribution_loss` selects `model/{causvid,dmd,sid}.py`. This is the main path; `trainer/ode.py` (ODE regression on LMDB pairs) and `trainer/gan.py` are the other two.
3. **`model/base.py`** — `BaseModel` builds the four networks: `generator` (causal, trainable), `real_score` and `fake_score` (both *bidirectional* Wan models; fake_score is the trainable critic), plus frozen `text_encoder` and `vae`. `SelfForcingModel._run_generator` is where the self-forcing rollout is invoked and where the gradient mask over frames is built.
4. **`pipeline/self_forcing_training.py`** — `SelfForcingTrainingPipeline.inference_with_trajectory`: the actual self-forcing loop. Per block it runs the multi-step denoising chain under `no_grad`, **backprops through exactly one randomly chosen denoising step** (`exit_flags`, broadcast from rank 0 so every rank picks the same step), then re-runs the block at `context_noise` timestep to write clean KV cache entries for the next block.

Inference mirrors this in `pipeline/causal_inference.py` (`CausalInferencePipeline`, few-step — chosen when the config has `denoising_step_list`) and `pipeline/causal_diffusion_inference.py` (multi-step). `pipeline/bidirectional_*.py` are the non-causal baselines.

`utils/wan_wrapper.py` is the boundary to the vendored Wan code: `WanDiffusionWrapper` (wraps `WanModel` or `CausalWanModel` depending on `is_causal`, owns the `FlowMatchScheduler`, and converts flow-prediction ↔ x0-prediction), `WanTextEncoder` (umT5-XXL), `WanVAEWrapper` (encode/decode plus a cached-decode path for streaming).

`wan/modules/causal_model.py` is the causal fork of `wan/modules/model.py`: block-wise causal attention masks, KV cache with rolling window (`local_attn_size`) and attention sink (`sink_size`, first N frames never evicted).

### Invariants baked into the code

The 1.3B model at 480×832 is hardcoded in many places — changing resolution or model size means touching all of these:

- 21 latent frames ↔ 81 pixel frames (4× temporal compression + 1). Latent shape `[B, 21, 16, 60, 104]`.
- `frame_seq_length = 1560` (= 60×104/4), `num_transformer_blocks = 30`, `seq_len = 32760` (= 21×1560), KV cache heads `[12, 128]`.
- `adding_cls_branch` in `WanDiffusionWrapper` is explicitly marked hardcoded for 1.3B.

### Frame-block semantics

`num_frame_per_block` (3 in the DMD config) is the chunk the model denoises jointly; `independent_first_frame` switches to a `[1, N, N, ...]` layout so the first latent frame can be image-conditioned (I2V). Both training and inference assert the frame count divides evenly into blocks, and both set `generator.model.num_frame_per_block` on the underlying module — a value set on the config alone will not take effect.

### Config layering

Every entry point does `OmegaConf.merge(configs/default_config.yaml, <your config>)`. `configs/default_config.yaml` holds the structural defaults (`causal`, `num_training_frames`, `independent_first_frame`, eval prompt paths); `configs/self_forcing_{dmd,sid}.yaml` hold the recipe. Note `warp_denoising_step: true` requires that `denoising_step_list` does *not* end in `0`.

### Checkpoint format

Trainers save `{"generator", "critic", "generator_ema"}` (the EMA key only appears after `ema_start_step`) to `<logdir>/checkpoint_model_%06d/model.pt`. `inference.py --use_ema` reads `generator_ema`; `demo.py` reads `generator_ema` unconditionally. Loading a pretrained generator via `config.generator_ckpt` tolerates a raw state dict or one nested under `generator`/`model`.

### Memory paths

Both `inference.py` and `demo.py` detect <40 GB free VRAM and switch on `demo_utils/memory.py`'s `DynamicSwapInstaller`, which swaps the text encoder module-by-module between CPU and GPU. `demo.py` additionally offers TAEHV VAE, FP8 linear layers, and `torch.compile` as speed/quality tradeoffs, all toggled at runtime from the web UI.

### Data

Training the DMD/SiD path is **data-free** — it consumes text prompts only (`prompts/vidprom_filtered_extended.txt`). Only ODE regression and GAN training need video data, as LMDB shards built by `scripts/create_lmdb_iterative.py` / `create_lmdb_14b_shards.py` from ODE pairs generated by `scripts/generate_ode_pairs.py`.

Prompts matter: the model was trained on long, detailed prompts and degrades on short ones. `TextDataset` supports a parallel `--extended_prompt_path` file for this reason.
