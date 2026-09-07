"""Self-Forcing T2V inference on Trainium: T5 -> causal DiT -> streaming VAE -> mp4.

Run with torchrun on a Neuron host, from the repo root, with the rolling_forcing
checkout on PYTHONPATH (see neuron/README.md):

    torchrun --nproc_per_node 8 neuron/sf_inference.py \
      --prompt_file prompts/MovieGenVideoBench_extended.txt \
      --config_path neuron/configs/self_forcing_dmd.yaml \
      --checkpoint_path checkpoints/self_forcing_dmd.pt \
      --output_folder videos_sf --use_ema

World size must be 8 (TP=4 x SP=2) at 480x832: frame_seq_length 1560 is
divisible by 8 but not 16, and the SP shard may not split a frame.
"""

import argparse
import datetime
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dit_pipeline import destroy_parallel_groups, init_parallel_groups  # rolling_forcing
from models.t5 import (
    build_text_encoder,
    destroy_t5_parallel_group,
    encode_one_prompt,
    init_t5_parallel_group,
)
from models.vae import build_vae, destroy_vae_parallel_group, init_vae_parallel_group
from utils import parallel_state as ps
from utils import w_shard
from utils.logging_utils import configure_logging, get_logger
from utils.video import save_video, video_tensor_to_uint8

from pipeline_self_forcing import build_sf_pipeline, probe

DEBUG_NUMERICS = os.environ.get("SF_DEBUG_NUMERICS", "0") == "1"

# "world": shard the VAE decode over W across all ranks (rolling_forcing's default,
# via init_vae_parallel_group registering the WORLD group).
# "none":  give every rank a singleton vae-sp group, so every group-dependent site
#          in models/vae.py takes its world == 1 fallback -- plain PyTorch conv,
#          no halo exchange, no extract_w_edges / restore_layout NKI kernels, no
#          all-gather. Each rank then decodes the full width redundantly.
VAE_SHARD = os.environ.get("SF_VAE_SHARD", "world")
assert VAE_SHARD in ("world", "none"), f"SF_VAE_SHARD={VAE_SHARD}"


def init_vae_group(rank, world):
    if VAE_SHARD == "world":
        init_vae_parallel_group()
        return world
    # new_group is collective: every rank must build every singleton, in order.
    singletons = [dist.new_group([r]) for r in range(world)]
    ps.register_group("vae-sp", singletons[rank])
    assert ps.get_world_size("vae-sp") == 1
    return 1


def vae_input(latent, rank, world):
    """The latent this rank feeds the decoder: its W shard, or the whole thing."""
    return latent if VAE_SHARD == "none" else w_shard(latent, rank, world)


# Write every frame as a PNG next to the mp4. Pillow is installed explicitly by the
# job rather than relied on as a transitive dep.
SAVE_FRAMES = os.environ.get("SF_SAVE_FRAMES", "1") == "1"


def gather_full_video(video_local, rank, world):
    """The whole frame on rank 0, None elsewhere.

    Unsharded, this rank already decoded full width. Sharded, the W shards go
    through files exactly as rolling_forcing's utils.video.gather_and_save does --
    these are CPU tensors and that is the idiom that stack uses for them.
    """
    if VAE_SHARD == "none":
        return video_local if rank == 0 else None

    scratch = os.path.join(tempfile.gettempdir(), "sf_vae_shards")
    if rank == 0:
        os.makedirs(scratch, exist_ok=True)
    dist.barrier()
    torch.save(video_local, os.path.join(scratch, f"shard_rank{rank}.pt"))
    dist.barrier()

    full = None
    if rank == 0:
        full = torch.cat(
            [torch.load(os.path.join(scratch, f"shard_rank{r}.pt"), map_location="cpu")
             for r in range(world)], dim=-1)
    dist.barrier()
    if rank == 0:
        for r in range(world):
            os.remove(os.path.join(scratch, f"shard_rank{r}.pt"))
        os.rmdir(scratch)
    return full


def save_frames(video, out_dir):
    """One PNG per frame. video is [B,T,C,H,W] in [-1,1]; video_tensor_to_uint8
    returns [B,T,H,W,C] uint8, which is already PIL's RGB layout."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    frames = video_tensor_to_uint8(video)[0]
    for i, frame in enumerate(frames):
        Image.fromarray(frame.numpy()).save(
            os.path.join(out_dir, f"frame_{i:04d}.png"))
    logger.info("  wrote %d frames to %s", frames.shape[0], out_dir)


configure_logging()
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Self-Forcing T2V inference on Neuron")
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--output_folder", type=str, default="videos_sf")
    p.add_argument("--num_output_frames", type=int, default=21,
                   help="Latent frames. 21 -> 81 pixel frames; the model was "
                        "distilled at 21 and does not extend past it.")
    p.add_argument("--tp_degree", type=int, default=4,
                   help="DiT tensor-parallel degree; sp = world / tp.")
    p.add_argument("--latent_h", type=int, default=60)
    p.add_argument("--latent_w", type=int, default=104)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Only generate the first N prompts of the file.")
    return p.parse_args()


def staggered(rank, world, waves, fn):
    """Run fn on a fraction of the ranks at a time, so host RAM peak is world/waves.

    Every rank torch.loads the full 11.4GB T5 encoder and the 5.7GB DiT checkpoint
    before sharding, so 16 ranks loading at once peaks near 270GB and OOMKilled the
    pod at 250Gi. Raising the request to 900Gi (what I did first) just monopolised the
    node and left no room for other jobs. Loading in waves fixes the cause: with 4
    waves the peak is a quarter, and the pod fits in the same 250Gi rolling_forcing
    asks for.
    """
    if waves <= 1:
        return fn()
    per_wave = (world + waves - 1) // waves
    out = None
    for w in range(waves):
        if rank // per_wave == w:
            out = fn()
        dist.barrier()          # every rank every wave, so this cannot deadlock
    return out


def vae_selftest(vae, rank, world, latent_h, latent_w, dtype, nfpb=3):
    """Decode latents the DiT never touched, through the exact production path.

    Decides where the non-finite pixels come from: if zeros and a plain Gaussian
    also come back non-finite, the width-sharded decode is broken independently of
    anything the denoising loop produces, and the DiT is exonerated. Shapes match
    the real chunks, so this reuses their NEFFs rather than compiling new ones.
    """
    logger.info("VAE self-test (latents the DiT never saw):")
    for name, latent in (
        ("zeros", torch.zeros(1, nfpb, 16, latent_h, latent_w, dtype=dtype)),
        ("gaussian", torch.randn(1, nfpb, 16, latent_h, latent_w, dtype=dtype)),
    ):
        vae.model.clear_cache()
        pixels = vae.postprocess_pixels(
            vae.decode_to_pixel_device(
                vae_input(latent.to("neuron"), rank, world),
                use_cache=True, chunk_idx=0))
        probe(f"selftest {name} pixels", pixels)
    vae.model.clear_cache()


def stream_decode_prompt(pipe, vae, prompt_embeds, noise, rank, world, fps):
    """Denoise block by block, decoding each block as soon as it is clean."""
    video_chunks = []
    block_ms = []
    t = time.perf_counter()

    for chunk_idx, chunk in enumerate(
            pipe.inference_self_forcing_stream(noise, {"prompt_embeds": prompt_embeds})):
        torch.neuron.synchronize()
        dit_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()

        chunk_device = vae.decode_to_pixel_device(
            vae_input(chunk, rank, world), use_cache=True, chunk_idx=chunk_idx)
        torch.neuron.synchronize()
        vae_ms = (time.perf_counter() - t) * 1000

        chunk_video = vae.postprocess_pixels(chunk_device)
        video_chunks.append(chunk_video)
        if DEBUG_NUMERICS:
            probe(f"chunk {chunk_idx} latent_in", chunk)
            probe(f"chunk {chunk_idx} pixels_out", chunk_video)

        frames = chunk_video.shape[1]
        total = dit_ms + vae_ms
        block_ms.append(total)
        logger.info("  block %2d: DiT %8.1f ms  VAE %7.1f ms  %2d frames  %5.2f fps",
                    chunk_idx, dit_ms, vae_ms, frames, frames * 1000.0 / total)
        t = time.perf_counter()

    video = torch.cat(video_chunks, dim=1)
    if block_ms:
        wall = sum(block_ms) / 1000.0
        logger.info("  video: %d frames in %.2f s -> %.2f fps (%.1f x realtime at %d fps)",
                    video.shape[1], wall, video.shape[1] / wall,
                    (video.shape[1] / wall) / fps, fps)
    return video, block_ms


def main():
    args = parse_args()

    os.environ.setdefault("NEURON_FALLBACK_ENABLED", "0")

    # The default collective timeout is 30 min, and a cold compile at a width the
    # cache has never seen blows straight through it: the watchdog kills a posted
    # allgather while another rank is still in neuronx-cc, and the run then retries
    # forever with exponential backoff instead of failing. Give compilation room.
    timeout_s = int(os.environ.get("SF_COLLECTIVE_TIMEOUT_S", "7200"))
    try:
        dist.init_process_group(
            backend="neuron",
            timeout=datetime.timedelta(seconds=timeout_s))
        logger.info("collective timeout: %d s", timeout_s)
    except TypeError:
        # Backend does not accept the kwarg on this SDK; do not fail over it.
        dist.init_process_group(backend="neuron")
        logger.warning("this backend ignores timeout=; using its default (~1800 s). "
                       "A long cold compile may trip the collective watchdog.")
    rank = dist.get_rank()
    world = dist.get_world_size()

    assert world % args.tp_degree == 0, (
        f"world_size {world} not divisible by tp_degree {args.tp_degree}")
    sp_degree = world // args.tp_degree

    init_t5_parallel_group()
    init_parallel_groups(sp_degree, args.tp_degree)
    vae_world = init_vae_group(rank, world)
    logger.info("VAE sharding: %s (vae-sp world %d)", VAE_SHARD, vae_world)

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    with open(args.prompt_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompt_file)

    waves = int(os.environ.get("SF_LOAD_WAVES", "4"))
    logger.info("Building T5 text encoder (%d load waves)...", waves)
    text_encoder = staggered(rank, world, waves,
                             lambda: build_text_encoder(device="neuron"))
    logger.info("Building Self-Forcing DiT pipeline (TP=%d SP=%d)...",
                args.tp_degree, sp_degree)
    pipe = staggered(rank, world, waves, lambda: build_sf_pipeline(
        args.config_path, args.checkpoint_path, args.tp_degree, args.use_ema))
    vae_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
        os.environ.get("SF_VAE_DTYPE", "bf16")]
    logger.info("Building VAE decoder (%s)...", vae_dtype)
    vae = build_vae(dtype=vae_dtype)

    if DEBUG_NUMERICS:
        vae_selftest(vae, rank, world, args.latent_h, args.latent_w, vae_dtype)

    logger.info("Schedule: %d frames / %d per block x (%d denoising steps + 1 cache pass) "
                "= %d model calls per video",
                args.num_output_frames, pipe.num_frame_per_block,
                len(pipe.denoising_step_list),
                (args.num_output_frames // pipe.num_frame_per_block)
                * (len(pipe.denoising_step_list) + 1))

    if rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    dist.barrier()

    timings = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info("[prompt %3d/%d] %s...", prompt_idx, len(prompts), prompt[:70])

        noise = torch.randn(
            1, args.num_output_frames, 16, args.latent_h, args.latent_w,
            dtype=torch.bfloat16,
        ).to("neuron")
        vae.model.clear_cache()

        torch.neuron.synchronize()
        t = time.perf_counter()
        prompt_embeds = encode_one_prompt(text_encoder, prompt)
        torch.neuron.synchronize()
        logger.info("  T5:        %8.1f ms", (time.perf_counter() - t) * 1000)
        if DEBUG_NUMERICS:
            probe("prompt_embeds", prompt_embeds)

        video_local, block_ms = stream_decode_prompt(
            pipe, vae, prompt_embeds, noise, rank, world, args.fps)
        timings.append((prompt_idx, video_local.shape[1], block_ms))

        finite = torch.isfinite(video_local).all().item()
        if not finite:
            # Under the debug flag, keep going: the mp4 still gets written and
            # archived, and one run then shows the whole picture instead of
            # dying at the first prompt.
            msg = f"prompt {prompt_idx} produced non-finite pixels on rank {rank}"
            if not DEBUG_NUMERICS:
                raise AssertionError(msg)
            logger.warning("%s (continuing: SF_DEBUG_NUMERICS=1)", msg)

        full = gather_full_video(video_local, rank, world)
        if rank == 0:
            out_path = os.path.join(args.output_folder, f"prompt_{prompt_idx:03d}.mp4")
            save_video(full, out_path, args.fps)
            if SAVE_FRAMES:
                save_frames(full, os.path.join(
                    args.output_folder, "frames", f"prompt_{prompt_idx:03d}"))
        dist.barrier()

    if rank == 0 and timings:
        logger.info("=== per-prompt timing ===")
        logger.info("The flash kernel traces one specialisation per distinct attention-"
                    "window length (%d of them here), and they are cached in-process. So "
                    "prompt 0 carries the tracing cost and later prompts do not: those "
                    "are the numbers to quote.",
                    args.num_output_frames // pipe.num_frame_per_block)
        for idx, frames, bm in timings:
            wall = sum(bm) / 1000.0
            tag = " (includes kernel tracing)" if idx == 0 else ""
            logger.info("  prompt %3d: %2d frames  %7.2f s  %5.2f fps  "
                        "block min %7.1f ms / median %7.1f ms%s",
                        idx, frames, wall, frames / wall,
                        min(bm), sorted(bm)[len(bm) // 2], tag)

    destroy_t5_parallel_group()
    destroy_parallel_groups()
    destroy_vae_parallel_group()
    dist.destroy_process_group()
    logger.info("Done.")


if __name__ == "__main__":
    main()
