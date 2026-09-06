"""Schedule gate: runs on a laptop, no Neuron and no torch needed.

    python neuron/tests/test_schedule.py

The expected sequence is written out literally rather than recomputed, so this
is a real check against ``pipeline/causal_inference.py`` and not a restatement
of ``neuron/schedule.py``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schedule import (  # noqa: E402
    CACHE,
    DENOISE,
    distinct_kv_window_lengths,
    sf_schedule,
    total_calls,
)

# The released chunk-wise DMD recipe: configs/self_forcing_dmd.yaml.
STEPS = [1000.0, 750.0, 500.0, 250.0]
NFPB = 3
NUM_FRAMES = 21


def expected_calls():
    """(kind, timestep, start_frame, step_index, renoise_to) in issue order.

    Read off pipeline/causal_inference.py: 7 blocks at start frames 0,3,...,18;
    within a block the 4 timesteps in list order, each re-noising to the next,
    then one call at context_noise=0 whose output is dropped.
    """
    per_block = [
        (DENOISE, 1000.0, 0, 750.0),
        (DENOISE, 750.0, 1, 500.0),
        (DENOISE, 500.0, 2, 250.0),
        (DENOISE, 250.0, 3, None),
        (CACHE, 0.0, None, None),
    ]
    out = []
    for start_frame in (0, 3, 6, 9, 12, 15, 18):
        for kind, timestep, step_index, renoise_to in per_block:
            out.append((kind, timestep, start_frame, step_index, renoise_to))
    return out


def test_sequence_matches_reference():
    got = [
        (c.kind, c.timestep, c.start_frame, c.step_index, c.renoise_to)
        for c in sf_schedule(NUM_FRAMES, NFPB, STEPS, context_noise=0.0)
    ]
    want = expected_calls()
    assert len(got) == 35, f"expected 35 calls, got {len(got)}"
    assert got == want, (
        "schedule diverged from pipeline/causal_inference.py; first mismatch: "
        + next(f"idx {i}: got {g} want {w}"
               for i, (g, w) in enumerate(zip(got, want)) if g != w))


def test_structural_invariants():
    calls = list(sf_schedule(NUM_FRAMES, NFPB, STEPS))

    # Blocks are issued in order and never revisited.
    starts = [c.start_frame for c in calls]
    assert starts == sorted(starts)
    for start in set(starts):
        assert starts.count(start) == 5, f"block at {start} got {starts.count(start)} calls"

    # The cache call is last within every block: exactly one per block, and the
    # call after it (if any) belongs to the next block.
    for i, c in enumerate(calls):
        if c.kind != CACHE:
            continue
        assert c.step_index is None
        if i + 1 < len(calls):
            assert calls[i + 1].start_frame == c.start_frame + NFPB

    # Every frame is covered exactly once, and the block always has nfpb frames.
    assert {c.num_frames for c in calls} == {NFPB}
    assert max(starts) + NFPB == NUM_FRAMES

    # Re-noising chains through the step list and stops at the last step.
    for c in calls:
        if c.kind == DENOISE and c.step_index < len(STEPS) - 1:
            assert c.renoise_to == STEPS[c.step_index + 1]
        else:
            assert c.renoise_to is None


def test_counts_and_windows():
    assert total_calls(NUM_FRAMES, NFPB, STEPS) == 35
    # Global attention: block b sees every frame generated so far.
    assert distinct_kv_window_lengths(NUM_FRAMES, NFPB) == [3, 6, 9, 12, 15, 18, 21]


def test_rejects_bad_shapes():
    for bad in (20, 22):
        try:
            list(sf_schedule(bad, NFPB, STEPS))
        except ValueError:
            pass
        else:
            raise AssertionError(f"num_frames={bad} should not be accepted")
    try:
        list(sf_schedule(NUM_FRAMES, NFPB, []))
    except ValueError:
        pass
    else:
        raise AssertionError("empty denoising_step_list should not be accepted")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall schedule gates passed")
