import sys

import pytest
import torch

from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
    commit_native_kda_replayssm_ppu,
)
from sglang.srt.utils.common import is_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not is_sail() or torch.cuda.get_device_capability() != (8, 9),
    reason="PPU SM89 kernel is required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_flashkda_replayssm_commit_accept_prefix_and_tracks(dtype) -> None:
    torch.manual_seed(29)
    layers, slots, heads, record_len, dim = 2, 8, 2, 4, 128
    state = (torch.randn(layers, slots, heads, dim, dim, device="cuda") * 0.05).to(
        dtype
    )
    correction = (
        torch.randn(layers, slots, heads, record_len, dim, device="cuda") * 0.01
    )
    key = torch.randn(layers, slots, heads, record_len, dim, device="cuda") * 0.05
    decay = (
        torch.rand(layers, slots, heads, record_len, dim, device="cuda")
        .mul_(0.05)
        .add_(0.95)
    )
    indices = torch.tensor([1, 2, 3, -1], device="cuda", dtype=torch.int32)
    accept_lens = torch.tensor([0, 1, record_len, 2], device="cuda", dtype=torch.int32)
    tracks = torch.tensor([-1, 5, 6, -1], device="cuda", dtype=torch.int32)
    steps = torch.tensor([-1, 0, 1, -1], device="cuda", dtype=torch.int32)

    initial_state = state.clone()
    expected_state = initial_state.clone()
    for slot, num_accept, track, track_step in zip(
        indices.tolist(),
        accept_lens.tolist(),
        tracks.tolist(),
        steps.tolist(),
    ):
        if slot < 0 or num_accept <= 0:
            continue
        current_state = initial_state[:, slot].float()
        for step in range(num_accept):
            current_state = (
                current_state * decay[:, slot, :, step, None, :]
                + correction[:, slot, :, step, :, None] * key[:, slot, :, step, None, :]
            )
            if track >= 0 and step == track_step:
                expected_state[:, track] = current_state.to(dtype)
        expected_state[:, slot] = current_state.to(dtype)

    commit_native_kda_replayssm_ppu(
        state,
        correction,
        key,
        decay,
        indices,
        accept_lens,
        tracks,
        steps,
    )
    torch.cuda.synchronize()

    tolerance = 0.01 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(state, expected_state, rtol=0, atol=tolerance)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
