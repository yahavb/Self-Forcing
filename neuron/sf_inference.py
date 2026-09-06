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
import os
import sys
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
from utils import w_shard
from utils.logging_utils import configure_logging, get_logger
from utils.video import gather_and_save

from pipeline_self_forcing import build_sf_pipeline, probe

DEBUG_NUMERICS = os.environ.get("SF_DEBUG_NUMERICS", "0") == "1"

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
            w_shard(chunk, rank, world), use_cache=True, chunk_idx=chunk_idx)
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
    return video


def main():
    args = parse_args()

    os.environ.setdefault("NEURON_FALLBACK_ENABLED", "0")
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()

    assert world % args.tp_degree == 0, (
        f"world_size {world} not divisible by tp_degree {args.tp_degree}")
    sp_degree = world // args.tp_degree

    init_t5_parallel_group()
    init_parallel_groups(sp_degree, args.tp_degree)
    init_vae_parallel_group()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    with open(args.prompt_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompt_file)

    logger.info("Building T5 text encoder...")
    text_encoder = build_text_encoder(device="neuron")
    logger.info("Building Self-Forcing DiT pipeline (TP=%d SP=%d)...",
                args.tp_degree, sp_degree)
    pipe = build_sf_pipeline(
        args.config_path, args.checkpoint_path, args.tp_degree, args.use_ema)
    logger.info("Building VAE decoder...")
    vae = build_vae(dtype=torch.bfloat16)

    logger.info("Schedule: %d frames / %d per block x (%d denoising steps + 1 cache pass) "
                "= %d model calls per video",
                args.num_output_frames, pipe.num_frame_per_block,
                len(pipe.denoising_step_list),
                (args.num_output_frames // pipe.num_frame_per_block)
                * (len(pipe.denoising_step_list) + 1))

    if rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    dist.barrier()

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

        video_local = stream_decode_prompt(
            pipe, vae, prompt_embeds, noise, rank, world, args.fps)

        finite = torch.isfinite(video_local).all().item()
        if not finite:
            # Under the debug flag, keep going: the mp4 still gets written and
            # archived, and one run then shows the whole picture instead of
            # dying at the first prompt.
            msg = f"prompt {prompt_idx} produced non-finite pixels on rank {rank}"
            if not DEBUG_NUMERICS:
                raise AssertionError(msg)
            logger.warning("%s (continuing: SF_DEBUG_NUMERICS=1)", msg)

        out_path = os.path.join(args.output_folder, f"prompt_{prompt_idx:03d}.mp4")
        gather_and_save(video_local, out_path, args.fps, rank, world)

    destroy_t5_parallel_group()
    destroy_parallel_groups()
    destroy_vae_parallel_group()
    dist.destroy_process_group()
    logger.info("Done.")


if __name__ == "__main__":
    main()
