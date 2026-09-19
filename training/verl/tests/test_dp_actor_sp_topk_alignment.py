# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from types import SimpleNamespace

from verl.workers.actor.dp_actor import _align_rmpad_topk_ids_for_ulysses


@pytest.mark.parametrize(
    ("sp_rank", "expected_rows"),
    [
        (0, [[0, 100], [1, 101], [2, 102]]),
        (1, [[3, 103], [4, 104], [0, 0]]),
    ],
)
def test_align_global_rmpad_topk_ids_like_ulysses_input_slice(sp_rank, expected_rows):
    # Five packed tokens are padded to six, then split into contiguous chunks
    # of three just like ulysses_pad_and_slice_inputs(..., sp_size=2).
    topk_ids = torch.tensor([[i, i + 100] for i in range(5)])

    local_ids = _align_rmpad_topk_ids_for_ulysses(
        topk_ids,
        local_token_count=3,
        padding_size=1,
        sp_size=2,
        sp_rank=sp_rank,
    )

    assert local_ids.tolist() == expected_rows
    assert local_ids.is_contiguous()


def test_align_accepts_ids_already_sliced_by_upstream():
    local_ids = torch.tensor([[7, 8], [9, 10], [11, 12]])

    aligned = _align_rmpad_topk_ids_for_ulysses(
        local_ids,
        local_token_count=3,
        padding_size=1,
        sp_size=2,
        sp_rank=1,
    )

    assert aligned.data_ptr() == local_ids.data_ptr()


@pytest.mark.parametrize(
    ("topk_ids", "kwargs", "match"),
    [
        (torch.zeros(1, 2, 3), {}, "must be \\[tokens, K\\]"),
        (torch.zeros(4, 2), {}, "do not align"),
        (torch.zeros(5, 2), {"sp_rank": 2}, "invalid Ulysses SP rank"),
    ],
)
def test_align_rejects_ambiguous_or_invalid_shapes(topk_ids, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _align_rmpad_topk_ids_for_ulysses(
            topk_ids,
            local_token_count=3,
            padding_size=1,
            sp_size=2,
            sp_rank=kwargs.get("sp_rank", 0),
        )


def test_trace_off_counts_real_optimizer_steps_and_nonfinite_skips(monkeypatch):
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    monkeypatch.setenv("MOPD_R5_TRACE", "0")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = SimpleNamespace(grad_clip=1.0)
    actor.actor_module = torch.nn.Linear(2, 1, bias=False)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    actor._r5_successful_optimizer_steps = 0
    actor._skipped_optimizer_steps = 0

    def unexpected_trace(*args):
        raise AssertionError("Trace-off optimizer must not create expensive samples")

    actor._r5_on_successful_step = unexpected_trace
    actor._r5_on_skipped_step = unexpected_trace
    before = actor.actor_module.weight.detach().clone()
    actor.actor_module.weight.grad = torch.ones_like(before)
    actor._optimizer_step()
    assert actor._r5_successful_optimizer_steps == 1
    assert not torch.equal(before, actor.actor_module.weight)
    after = actor.actor_module.weight.detach().clone()
    actor.actor_module.weight.grad = torch.full_like(after, float("nan"))
    actor._optimizer_step()
    assert actor._r5_successful_optimizer_steps == 1
    assert actor._skipped_optimizer_steps == 1
    assert torch.equal(after, actor.actor_module.weight)
