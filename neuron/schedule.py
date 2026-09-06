"""Self-Forcing's chunk-wise causal inference schedule, as plain data.

Kept free of torch (and of any Neuron import) so it can be unit-tested on a
laptop. The runtime pipeline walks this schedule and issues one model call per
entry; see ``neuron/pipeline_self_forcing.py``.

Reference: ``pipeline/causal_inference.py`` in this repo, the temporal denoising
loop (Step 3). Per block of ``num_frame_per_block`` latent frames it runs the
whole ``denoising_step_list`` at a single uniform timestep per step, re-noising
to the next timestep in between, then a final call at ``context_noise`` whose
output is discarded and whose only job is to leave clean K/V in the block's
cache slots.
"""

from typing import Iterator, List, NamedTuple, Optional, Sequence

DENOISE = "denoise"
CACHE = "cache"


class SFCall(NamedTuple):
    """One model call.

    kind:        DENOISE (its x0 prediction is used) or CACHE (output discarded).
    timestep:    uniform timestep for every frame in the block.
    block_index: 0-based block counter.
    start_frame: first latent frame of the block, absolute.
    num_frames:  frames in the block (always num_frame_per_block).
    step_index:  index into denoising_step_list, or None for the CACHE call.
    renoise_to:  timestep to re-noise the x0 prediction to before the next call,
                 or None when this is the last denoising step (nothing to
                 re-noise) or a CACHE call.
    """

    kind: str
    timestep: float
    block_index: int
    start_frame: int
    num_frames: int
    step_index: Optional[int]
    renoise_to: Optional[float]


def sf_schedule(
    num_frames: int,
    num_frame_per_block: int,
    denoising_step_list: Sequence[float],
    context_noise: float = 0.0,
) -> Iterator[SFCall]:
    """Yield the model calls for one video, in issue order.

    ``denoising_step_list`` must already be warped if the config asks for it
    (``warp_denoising_step``) — this function does not interpret the values, it
    only orders them.
    """
    if num_frames % num_frame_per_block != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be a multiple of "
            f"num_frame_per_block ({num_frame_per_block})")
    if not denoising_step_list:
        raise ValueError("denoising_step_list must be non-empty")

    steps = list(denoising_step_list)
    num_blocks = num_frames // num_frame_per_block

    for block_index in range(num_blocks):
        start_frame = block_index * num_frame_per_block
        for step_index, timestep in enumerate(steps):
            is_last = step_index == len(steps) - 1
            yield SFCall(
                kind=DENOISE,
                timestep=timestep,
                block_index=block_index,
                start_frame=start_frame,
                num_frames=num_frame_per_block,
                step_index=step_index,
                renoise_to=None if is_last else steps[step_index + 1],
            )
        yield SFCall(
            kind=CACHE,
            timestep=context_noise,
            block_index=block_index,
            start_frame=start_frame,
            num_frames=num_frame_per_block,
            step_index=None,
            renoise_to=None,
        )


def calls_per_block(denoising_step_list: Sequence[float]) -> int:
    """Denoising calls plus the one clean-context cache call."""
    return len(denoising_step_list) + 1


def total_calls(num_frames: int, num_frame_per_block: int,
                denoising_step_list: Sequence[float]) -> int:
    return (num_frames // num_frame_per_block) * calls_per_block(denoising_step_list)


def distinct_kv_window_lengths(num_frames: int, num_frame_per_block: int) -> List[int]:
    """Attention-window lengths in frames, one per block.

    Self-Forcing attends over every frame generated so far, so block b sees
    (b+1)*num_frame_per_block frames. Each distinct length is a separate
    ``actual_seqlen_k`` into the flash-attention kernel, hence a separate NKI
    specialisation — this is what the first run spends its compile time on.
    """
    num_blocks = num_frames // num_frame_per_block
    return [(b + 1) * num_frame_per_block for b in range(num_blocks)]
