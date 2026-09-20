"""CPU-only E08 contract through native forward, loss and optimizer loops."""
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from verl.workers.actor import dp_actor


class CausalLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.linspace(-1, 1, 32))

    def forward(self, input_ids, **kwargs):
        prefix = input_ids.cumsum(-1).float().unsqueeze(-1) / 100
        return SimpleNamespace(logits=self.values.view(1, 1, -1) + prefix)


def run_window(tmp_path, monkeypatch, epochs, corrupt=None):
    monkeypatch.setenv("MOPD_R5_TRACE", "1")
    monkeypatch.setenv("MOPD_R5_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(dp_actor, "_r5_rank", lambda: 0)
    actor = dp_actor.DataParallelPPOActor.__new__(dp_actor.DataParallelPPOActor)
    actor.config = OmegaConf.create({
        "grad_clip": 1.0, "kl_loss_coef": 0.0, "use_kl_loss": False,
        "ppo_mini_batch_size": 2, "ppo_micro_batch_size_per_gpu": 1,
        "ppo_epochs": epochs, "use_dynamic_bsz": False, "entropy_coeff": 0.0,
        "loss_agg_mode": "token-mean", "policy_loss": {"loss_mode": "vanilla"},
        "clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.28,
    })
    actor.actor_module = CausalLogits()
    actor.actor_optimizer = torch.optim.Adam(actor.actor_module.parameters(), lr=1.5e-6)
    actor.device_name = "cpu"
    actor.use_remove_padding = actor.use_fused_kernels = False
    actor._opd_refresh_advantage = True
    actor._opd_reward_weight_mode = "student_p"
    actor._r5_support_outer_captured = False
    actor._r5_successful_optimizer_steps = actor._skipped_optimizer_steps = 0
    scheduler = get_constant_schedule_with_warmup(actor.actor_optimizer, 0)
    teacher = torch.log_softmax(torch.linspace(1, -1, 32), -1)
    acquired, anchors, lr_at_updates = [], [], []
    actor.actor_optimizer.register_step_pre_hook(
        lambda optimizer, args, kwargs: lr_at_updates.append(optimizer.param_groups[0]["lr"])
    )
    for pool in range(4 // epochs):
        ids = torch.arange(16).view(1, 1, 16).expand(8, 3, 16).clone()
        old = torch.log_softmax(actor.actor_module.values.detach(), -1)[ids]
        q = teacher[ids].clone()
        data = DataProto.from_dict(tensors={
            "responses": torch.arange(24).reshape(8, 3),
            "response_mask": torch.tensor([[1, 1, 0]] * 8),
            "input_ids": torch.arange(40).reshape(8, 5),
            "attention_mask": torch.ones(8, 5, dtype=torch.long),
            "position_ids": torch.arange(5).repeat(8, 1),
            "old_log_probs": torch.full((8, 3), -4.0),
            "student_top_k_ids": ids, "student_top_k_log_probs": old.clone(),
            "teacher_on_student_log_probs": q,
            "advantages": torch.zeros(8, 3, 16),
            "domain_loss_weight": torch.arange(1, 9, dtype=torch.float32),
        }, non_tensors={
            "uid": np.array([f"pool{pool}-row{i}" for i in range(8)], dtype=object),
            "data_source": np.array(["math", "code", "if", "math"] * 2, dtype=object),
        }, meta_info={"temperature": 1.0})
        data.reorder(torch.tensor([7, 0, 6, 1, 5, 2, 4, 3]))
        acquired.append(pool)
        frozen = {k: data.batch[k].clone() for k in ("student_top_k_ids", "teacher_on_student_log_probs", "student_top_k_log_probs", "old_log_probs")}
        if corrupt == "support":
            data.batch["student_top_k_ids"] += 16  # Incorrect newly selected top16.
        elif corrupt == "q":
            data.batch["teacher_on_student_log_probs"] += 0.5  # Unintended rescore.
        metrics = actor.update_policy(data)
        assert metrics["actor/successful_optimizer_steps_this_outer"] == [4 * epochs]
        assert metrics["actor/skipped_optimizer_steps_this_outer"] == [0]
        scheduler.step()  # Native scheduler advances once per outer, not per update.
        for key, expected in frozen.items():
            torch.testing.assert_close(data.batch[key], expected)
        anchors.append(old)
    records = [json.loads(x) for x in (tmp_path / "r5_trace/optimizer_steps.rank0.jsonl").read_text().splitlines()]
    steps = [x for x in records if x["event"] == "successful_optimizer_step"]
    samples = [x for x in records if x["event"] == "learner_fixed_support"]
    assert [x["successful_optimizer_step"] for x in steps] == list(range(1, 17))
    assert len(acquired) == (4 if epochs == 1 else 1)
    assert lr_at_updates == [1.5e-6] * 16
    assert scheduler.last_epoch == len(acquired)
    for sample in samples:
        assert sample["student_top_k_ids"] == [list(range(16))] * 2
        torch.testing.assert_close(torch.tensor(sample["teacher_on_student_log_probs"]), teacher[:16].repeat(2, 1))
        s = torch.tensor(sample["current_student_top_k_log_probs"])
        q = torch.tensor(sample["teacher_on_student_log_probs"])
        expected = (q - s) * torch.softmax(s, -1)
        torch.testing.assert_close(torch.tensor(sample["advantage_before_domain_weight"]), expected)
        row = int(sample["sample_id"].split("row")[-1])
        torch.testing.assert_close(torch.tensor(sample["advantage_for_loss"]), expected * (row + 1))
    if epochs == 4:
        assert set(Counter(x["sample_id"] for x in samples).values()) == {4}
        assert samples[0]["current_student_top_k_log_probs"] != samples[4]["current_student_top_k_log_probs"]
    else:
        assert not torch.equal(anchors[0], anchors[-1])


@pytest.mark.parametrize("epochs", [1, 4])
def test_matched16_updates_keep_original_q_support_and_refresh_student(tmp_path, monkeypatch, epochs):
    run_window(tmp_path, monkeypatch, epochs)


@pytest.mark.parametrize("corrupt", ["support", "q"])
def test_window_oracle_rejects_new_support_or_rescore(tmp_path, monkeypatch, corrupt):
    with pytest.raises(AssertionError):
        run_window(tmp_path, monkeypatch, 4, corrupt)
