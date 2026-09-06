"""Self-Forcing chunk-wise causal inference on Neuron.

Rides on the rolling_forcing Neuron stack (``models/``, ``kernels/``,
``utils/`` — see neuron/README.md for the commit pin): same Wan2.1-T2V-1.3B
causal DiT, same NKI kernels, same TP/SP sharding. Only the denoising schedule
is Self-Forcing's.
"""

import torch
from omegaconf import OmegaConf

from models.dit_pipeline import (  # rolling_forcing
    CausalInferencePipeline,
    _shard_full_state_dict,
)
from utils.logging_utils import get_logger

from schedule import CACHE, DENOISE, sf_schedule

logger = get_logger(__name__)


class SelfForcingInferencePipeline(CausalInferencePipeline):
    """Block-at-a-time causal inference with a global KV window.

    Inherits model construction, KV/cross-attention cache allocation, the
    scheduler and ``_timestep_to_sigma`` from the rolling-forcing pipeline, and
    replaces only the loop. Per block of ``num_frame_per_block`` frames:

      1. ``len(denoising_step_list)`` forwards at a single uniform timestep each,
         re-noising the x0 prediction to the next timestep in between;
      2. one forward at ``context_noise`` whose output is discarded and whose
         only effect is to leave clean K/V in this block's cache slots.

    Every forward passes ``updating_cache=True``. That matters: it selects the
    branch of ``_assemble_kv`` (rolling_forcing/models/dit_attention.py:420)
    that builds the attention window from the *whole* cache with the unroped
    anchor re-RoPEd at absolute frame 0 — i.e. global attention over frames
    [0, block_end), which is what the released Self-Forcing checkpoints were
    distilled with (``local_attn_size=-1``). The ``updating_cache=False`` branch
    is the rolling-window one: it caps the carried history at
    ``max_attention_size - valid_tokens - block_length`` and re-RoPEs the anchor
    to a *relative* position, both wrong for this model.

    ``_cache_write`` is idempotent at a fixed ``cache_start``: the second and
    later calls for a block see ``num_new_tokens == 0``, so they rewrite the same
    slots without advancing ``global_end_index``/``local_end_index``. That is
    what makes the 4 noisy writes + 1 clean overwrite land on one set of slots,
    exactly as the reference implementation intends.
    """

    def _sf_asserts(self, num_frames, height, width):
        nfpb = self.num_frame_per_block
        attn = self.generator.model.blocks[0].self_attn

        assert self.num_frame_per_block == 3, (
            f"num_frame_per_block={nfpb}, but the Neuron attention hardcodes "
            f"block_length = 3 * frame_length "
            f"(rolling_forcing/models/dit_attention.py:81)")
        assert self.local_attn_size == -1, (
            f"local_attn_size={self.local_attn_size}; Self-Forcing's released "
            f"checkpoints use global attention. A windowed checkpoint needs the "
            f"updating_cache=False path instead.")

        max_frames = attn.max_attention_size // self.frame_seq_length
        assert num_frames <= max_frames, (
            f"num_frames={num_frames} exceeds the attention window of "
            f"{max_frames} frames (max_attention_size={attn.max_attention_size}). "
            f"Self-Forcing is a {max_frames}-latent-frame model; long rollouts "
            f"are what rolling forcing exists for.")

        # No eviction may fire: the sink/roll path in _cache_write assumes the
        # rolling window and would corrupt a global-attention run.
        cache_frames = attn.kv_cache_logical_size // self.frame_seq_length
        assert num_frames <= cache_frames, (
            f"num_frames={num_frames} > kv cache capacity {cache_frames} frames; "
            f"eviction would fire")

        # frame_seq_length drives cache slicing, RoPE indexing and the SP shard.
        # A mismatch corrupts silently instead of failing, so check it here.
        _pT, pH, pW = self.generator.model.patch_size
        runtime_fs = (height // pH) * (width // pW)
        assert runtime_fs == self.frame_seq_length, (
            f"frame_seq_length={self.frame_seq_length} (config) != runtime "
            f"h*w={runtime_fs} (latent {height}x{width} / patch {pH}x{pW}); "
            f"config and --latent_h/--latent_w must agree")
        assert self.frame_seq_length % self.world_size == 0, (
            f"frame_seq_length={self.frame_seq_length} not divisible by "
            f"world_size={self.world_size}; the SP shard would split a frame. "
            f"1560 (480x832) works at world 8, not 16.")

    def _reset_caches(self, batch_size, dtype, device):
        if self.kv_cache_clean is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
            return
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False
        for block_index in range(len(self.kv_cache_clean)):
            self.kv_cache_clean[block_index]["global_end_index"] = 0
            self.kv_cache_clean[block_index]["local_end_index"] = 0

    @torch.no_grad()
    def inference_self_forcing(self, noise, conditional_dict):
        final = None
        for chunk in self._run(noise, conditional_dict, streaming=False):
            final = chunk
        return final

    @torch.no_grad()
    def inference_self_forcing_stream(self, noise, conditional_dict):
        """Yields one clean block of ``num_frame_per_block`` latent frames at a time."""
        yield from self._run(noise, conditional_dict, streaming=True)

    def _run(self, noise, conditional_dict, streaming):
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert batch_size == 1, f"batch size must be 1, got {batch_size}"
        nfpb = self.num_frame_per_block
        assert num_frames % nfpb == 0, (
            f"num_frames={num_frames} must be a multiple of {nfpb}")
        self._sf_asserts(num_frames, height, width)

        device, dtype = noise.device, noise.dtype
        self._reset_caches(batch_size, dtype, device)

        # One shape for all calls: every forward in the run reuses these.
        block_input = torch.zeros(
            [batch_size, nfpb, num_channels, height, width], device=device, dtype=dtype)
        block_clean = torch.zeros_like(block_input)
        block_timestep = torch.zeros([batch_size, nfpb], device=device, dtype=torch.float32)
        block_sigma = torch.zeros([batch_size, nfpb], device=device, dtype=torch.float32)
        shared_buffers = (self.shared_buffer_k, self.shared_buffer_v)

        output = None if streaming else torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=device, dtype=dtype)

        steps = [t.item() for t in self.denoising_step_list]
        sigma_of = {t: self._timestep_to_sigma(t) for t in steps}
        # add_noise wants sigma broadcastable over the flattened [B*F, C, H, W].
        renoise_sigma = {
            t: sigma_of[t] * torch.ones([batch_size * nfpb, 1, 1, 1],
                                        device=device, dtype=torch.float32)
            for t in steps
        }

        pred_x0 = None
        for call in sf_schedule(num_frames, nfpb, steps, self.context_noise):
            if call.kind == DENOISE:
                if call.step_index == 0:
                    block_input.copy_(
                        noise[:, call.start_frame:call.start_frame + nfpb])
                block_timestep.fill_(call.timestep)
                block_sigma.fill_(sigma_of[call.timestep])
                _, pred_x0 = self.generator(
                    noisy_image_or_video=block_input,
                    conditional_dict=conditional_dict,
                    timestep=block_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=call.start_frame * self.frame_seq_length,
                    num_valid_frames=nfpb,
                    shared_buffers=shared_buffers,
                    sigma=block_sigma,
                    mode="denoise",
                    updating_cache=True,
                )
                if call.renoise_to is not None:
                    fresh = torch.randn(
                        [batch_size * nfpb, num_channels, height, width],
                        dtype=dtype).to(device)
                    block_input.copy_(
                        self._add_noise(
                            pred_x0.flatten(0, 1), fresh,
                            renoise_sigma[call.renoise_to],
                        ).unflatten(0, (batch_size, nfpb)))
                continue

            # CACHE call: rewrite this block's cache slots with clean context.
            block_clean.copy_(pred_x0)
            block_timestep.fill_(call.timestep)
            block_sigma.fill_(self.context_sigma)
            self.generator(
                noisy_image_or_video=block_clean,
                conditional_dict=conditional_dict,
                timestep=block_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=call.start_frame * self.frame_seq_length,
                num_valid_frames=nfpb,
                shared_buffers=shared_buffers,
                sigma=block_sigma,
                mode="denoise",
                updating_cache=True,
            )
            if streaming:
                yield block_clean.clone()
            else:
                output[:, call.start_frame:call.start_frame + nfpb].copy_(block_clean)

        if not streaming:
            yield output


def build_sf_pipeline(config_path, checkpoint_path, tp_degree, use_ema):
    """Mirror of rolling_forcing's build_dit_pipeline for the SF schedule.

    Not reusing that function directly because it merges a
    ``configs/default_config.yaml`` resolved relative to the *rolling_forcing*
    working directory; neuron/configs/self_forcing_dmd.yaml is self-contained.
    """
    config = OmegaConf.load(config_path)
    assert hasattr(config, "denoising_step_list"), (
        f"{config_path} has no denoising_step_list; the few-step (distilled) "
        f"checkpoint is the only one this pipeline runs")

    pipe = SelfForcingInferencePipeline(
        denoising_step_list=config.denoising_step_list,
        num_frame_per_block=getattr(config, "num_frame_per_block", 3),
        context_noise=getattr(config, "context_noise", 0.0),
        warp_denoising_step=getattr(config, "warp_denoising_step", True),
        model_name=getattr(config, "model_name", "Wan2.1-T2V-1.3B"),
        timestep_shift=getattr(config, "timestep_shift", 5.0),
        frame_seq_length=getattr(config, "frame_seq_length", 1560),
        local_attn_size=getattr(config, "local_attn_size", -1),
        sink_size=getattr(config, "sink_size", 0),
    )

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        key = "generator_ema" if use_ema else "generator"
        assert key in state_dict, (
            f"{checkpoint_path} has no '{key}' (keys: {sorted(state_dict)}); "
            f"--use_ema selects generator_ema")
        to_load = {
            k.replace("_fsdp_wrapped_module.", ""): v
            for k, v in state_dict[key].items()
        }

        if tp_degree > 1:
            to_load = _shard_full_state_dict(
                to_load,
                len(pipe.generator.model.blocks),
                pipe.generator.model.dim,
                pipe.generator.model.num_heads,
            )

        # torch.compile wraps submodules, so checkpoint keys need an _orig_mod
        # segment inserted at whichever level the wrapper sits.
        model_keys = set(pipe.generator.state_dict().keys())
        remapped = {}
        for k, v in to_load.items():
            if k in model_keys:
                remapped[k] = v
                continue
            parts = k.split(".")
            for i in range(len(parts)):
                cand = ".".join(parts[:i + 1] + ["_orig_mod"] + parts[i + 1:])
                if cand in model_keys:
                    remapped[cand] = v
                    break
            else:
                remapped[k] = v
        # assign=True: the model is built on the meta device via from_config, so a
        # copy-based load is a silent no-op and the later .to("neuron") fails with
        # "Cannot copy out of meta tensor".
        pipe.generator.load_state_dict(remapped, strict=True, assign=True)

    # assign=True loaded checkpoint dtypes; cast so weights match the bf16
    # activations, else the Neuron matmul compile fails on mismatched dtypes.
    pipe.generator.model = pipe.generator.model.to(device="neuron", dtype=torch.bfloat16)
    return pipe
