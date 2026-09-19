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

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


@pytest.mark.parametrize("refresh", [False, True])
def test_learner_trace_keeps_same_row_support_through_later_minibatches(tmp_path, monkeypatch, refresh):
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.workers.actor import dp_actor
    from verl.workers.actor.mt_opd import refresh_opd_advantage

    monkeypatch.setenv("MOPD_R5_TRACE", "1")
    monkeypatch.setenv("MOPD_R5_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(dp_actor, "_r5_rank", lambda: 0)
    monkeypatch.setattr(dp_actor, "get_policy_loss_fn", lambda mode: lambda **kw: (
        -(kw["log_prob"] * kw["advantages"]).mean(), {}
    ))
    actor = dp_actor.DataParallelPPOActor.__new__(dp_actor.DataParallelPPOActor)
    actor.config = OmegaConf.create({
        "grad_clip": 1.0, "kl_loss_coef": 0.0, "use_kl_loss": False,
        "ppo_mini_batch_size": 2, "ppo_micro_batch_size_per_gpu": 1, "ppo_epochs": 1,
        "use_dynamic_bsz": False, "entropy_coeff": 0.0, "loss_agg_mode": "token-mean",
        "policy_loss": {"loss_mode": "vanilla"},
    })
    actor.actor_module = torch.nn.Linear(1, 1, bias=False)
    actor.actor_module.weight.data.fill_(0.1)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.05)
    actor._r5_successful_optimizer_steps = 0
    actor._skipped_optimizer_steps = 0
    actor._opd_refresh_advantage = refresh
    actor._opd_reward_weight_mode = "student_p"
    actor._r5_support_outer_captured = False
    actor._r5_snapshot_params = lambda: {}
    actor._r5_write_param_delta = lambda: None
    expected = []

    def forward(inputs, *, student_top_k_ids, **kwargs):
        assert torch.equal(student_top_k_ids, inputs["student_top_k_ids"])
        current = inputs["student_top_k_log_probs"] + actor.actor_module.weight.reshape(1, 1, 1)
        advantage = refresh_opd_advantage(
            student_top_k_log_probs=current,
            teacher_on_student_log_probs=inputs["teacher_on_student_log_probs"],
            response_mask=inputs["response_mask"], reward_weight_mode="student_p",
        ) if refresh else inputs["advantages"]
        expected.append((str(inputs.get("uid", [""])[0]), current.detach().clone(), advantage.detach().clone()))
        return None, current[..., 0], student_top_k_ids, current

    actor._forward_micro_batch = forward
    ids = torch.arange(8 * 4 * 16).reshape(8, 4, 16)
    tensors = {
        "responses": torch.arange(32).reshape(8, 4),
        "response_mask": torch.tensor([[1, 1, 1, 0]] * 8),
        "input_ids": torch.ones(8, 6, dtype=torch.long),
        "attention_mask": torch.ones(8, 6, dtype=torch.long),
        "position_ids": torch.arange(6).repeat(8, 1),
        "old_log_probs": torch.full((8, 4), -3.0),
        "student_top_k_ids": ids,
        "student_top_k_log_probs": -3.0 - ids.float() / 1000,
        "advantages": torch.ones(8, 4, 16),
        "domain_loss_weight": torch.arange(1, 9, dtype=torch.float32),
    }
    if refresh:
        tensors["teacher_on_student_log_probs"] = -2.0 - ids.float() / 1000
    data = DataProto.from_dict(tensors=tensors, non_tensors={
        "uid": np.array([f"row-{i}" for i in range(8)], dtype=object),
        "data_source": np.array(["math", "code", "if", "math"] * 2, dtype=object),
    }, meta_info={"temperature": 1.0})
    metrics = actor.update_policy(data)
    path = tmp_path / "r5_trace/optimizer_steps.rank0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    samples = [row for row in records if row["event"] == "learner_fixed_support"]
    learner = json.loads((tmp_path / "r5_trace/learner_input.rank0.json").read_text())
    assert learner["has_teacher_on_student_log_probs"] is refresh
    assert ("teacher_on_student_log_probs" in learner["batch_keys"]) is refresh
    assert len(samples) == 4
    assert metrics["actor/successful_optimizer_steps_this_outer"] == [4]
    for mini_index, row in enumerate(samples):
        sample_index = 2 * mini_index
        uid, current, advantage = expected[sample_index]
        assert row["sample_id"] == uid
        assert row["data_source"] == data.non_tensor_batch["data_source"][sample_index]
        assert row["minibatch_index"] == mini_index
        assert row["successful_optimizer_steps_before"] == mini_index
        assert row["response_positions"] == [0, 1, 2]
        assert row["response_mask"] == [1, 1, 1]
        assert row["student_top_k_ids"] == ids[sample_index, :3].tolist()
        assert row["current_student_top_k_log_probs"] == current[0, :3].tolist()
        assert row["advantage_before_domain_weight"] == advantage[0, :3].tolist()
        torch.testing.assert_close(torch.tensor(row["advantage_for_loss"]), advantage[0, :3] * (sample_index + 1))
        assert (row["teacher_on_student_log_probs"] is not None) is refresh
        if refresh:
            assert row["teacher_on_student_log_probs"] == tensors["teacher_on_student_log_probs"][sample_index, :3].tolist()
    successful = [row for row in records if row["event"] == "successful_optimizer_step"]
    if refresh:
        for mini_index, row in enumerate(successful):
            assert row["teacher_q_sha256"] == dp_actor._r5_tensor_sha256(tensors["teacher_on_student_log_probs"][2 * mini_index:2 * mini_index + 2])
    actor.update_policy(data)
    assert sum(json.loads(line)["event"] == "learner_fixed_support" for line in path.read_text().splitlines()) == 4
    trace_before = path.read_text()
    monkeypatch.setenv("MOPD_R5_TRACE", "0")
    metrics = actor.update_policy(data)
    assert metrics["actor/successful_optimizer_steps_this_outer"] == [4]
    assert path.read_text() == trace_before
