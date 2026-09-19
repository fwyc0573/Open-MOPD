# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Regression tests for explicit teacher response-position alignment."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import (
    RewardModelWorker,
    _build_token_level_teacher_inputs,
    _validate_token_level_tokenizers,
)


class _Tokenizer:
    def __init__(self, *, eos_token_id=2, pad_token_id=0, prompt_ids=(90, 91, 92), vocab_shift=0):
        self.eos_token_id = eos_token_id
        self.bos_token_id = 1
        self.pad_token_id = pad_token_id
        self._prompt_ids = prompt_ids
        self._vocab = {f"token-{i}": i + vocab_shift for i in range(100)}

    def __len__(self):
        return 100

    def get_vocab(self):
        return self._vocab

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is True
        assert messages
        return list(self._prompt_ids)


class _CausalModel:
    """Position-sensitive causal logits, with packed sequence boundaries from positions."""

    def __call__(self, input_ids, attention_mask, position_ids, **kwargs):
        weights = torch.arange(100, dtype=torch.float64)
        logits = torch.empty((*input_ids.shape, 100), dtype=torch.float64)
        for row in range(input_ids.shape[0]):
            state = 0
            for pos in range(input_ids.shape[1]):
                if position_ids[row, pos] == 0:
                    state = 0
                if attention_mask is None or attention_mask[row, pos]:
                    state += int(input_ids[row, pos])
                logits[row, pos] = -((weights - (state % 97)) / 13).square()
        return SimpleNamespace(logits=logits)


def _native_scores(rows, *, remove_padding=False, candidates=None):
    worker = object.__new__(RewardModelWorker)
    worker.reward_module = _CausalModel()
    worker.use_remove_padding = remove_padding
    worker.use_fused_kernels = False
    worker.use_ulysses_sp = False
    batch = {name: torch.stack([row[index] for row in rows]) for index, name in enumerate(
        ("input_ids", "attention_mask", "responses")
    )}
    batch["position_ids"] = compute_position_id_with_mask(batch["attention_mask"])
    batch_size, width = batch["responses"].shape
    if candidates is None:
        candidates = (torch.arange(batch_size * width * 16).view(batch_size, width, 16) * 7 + 3) % 100
    scores = worker._forward_micro_batch(
        batch, student_top_k_ids=candidates, compute_entropy=True, top_k=16, strategy="only_stu"
    )
    return scores, candidates


def _assert_native_reference(row, scores, candidates, *, prompt_ids, valid_ids):
    """Reference starts with source prompt/response, never reconstructed builder output."""
    reference_ids = torch.tensor([list(prompt_ids) + list(valid_ids)])
    reference_logits = _CausalModel()(
        reference_ids, torch.ones_like(reference_ids), torch.arange(reference_ids.shape[1]).unsqueeze(0)
    ).logits[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(valid_ids)]
    log_probs = reference_logits.log_softmax(-1)
    torch.testing.assert_close(scores[0][row, : len(valid_ids)], log_probs.gather(
        -1, torch.tensor(valid_ids).unsqueeze(-1)
    ).squeeze(-1))
    torch.testing.assert_close(scores[1][row, : len(valid_ids)], log_probs.gather(
        -1, candidates[row, : len(valid_ids)]
    ))
    torch.testing.assert_close(scores[4][row, : len(valid_ids)], -(log_probs.exp() * log_probs).sum(-1))
    teacher_ids = reference_logits.topk(16, dim=-1).indices
    assert torch.equal(scores[2][row, : len(valid_ids)], teacher_ids)
    torch.testing.assert_close(scores[3][row, : len(valid_ids)], log_probs.gather(-1, teacher_ids))
    assert torch.equal(scores[5][row, : len(valid_ids)], torch.full((len(valid_ids),), 16))
    matches = candidates[row, : len(valid_ids)].unsqueeze(-1) == teacher_ids.unsqueeze(-2)
    assert torch.equal(scores[6][row, : len(valid_ids)].bool(), matches.any(-1))
    assert torch.equal(scores[7][row, : len(valid_ids)].bool(), matches.any(-2))


def test_short_response_through_production_slice_and_fixed_student_gather():
    row = _build_token_level_teacher_inputs(
        raw_prompt=[{"role": "user", "content": "question"}],
        source_response_ids=torch.tensor([10, 11, 0, 0, 0, 0]),
        response_mask=torch.tensor([1, 1, 0, 0, 0, 0]),
        source_tokenizer=_Tokenizer(), target_tokenizer=_Tokenizer(), response_length=6, max_length=10,
    )
    scores, candidates = _native_scores([row])
    _assert_native_reference(0, scores, candidates, prompt_ids=[90, 91, 92], valid_ids=[10, 11])
    assert row[0].tolist() == [0, 90, 91, 92, 10, 11, 0, 0, 0, 0]


def test_teacher_sequence_preserves_the_original_response_slots() -> None:
    source = _Tokenizer()
    target = _Tokenizer()
    input_ids, attention_mask, responses, valid_length = _build_token_level_teacher_inputs(
        raw_prompt=[{"role": "user", "content": "question"}],
        source_response_ids=torch.tensor([10, 11, 12, 13]),
        response_mask=torch.tensor([1, 1, 1, 0]),
        source_tokenizer=source,
        target_tokenizer=target,
        response_length=4,
        max_length=8,
    )

    assert valid_length == 3
    assert attention_mask.tolist() == [0, 1, 1, 1, 1, 1, 1, 0]
    assert responses.tolist() == [10, 11, 12, 13]
    assert torch.equal(input_ids[-4:], responses)
    assert compute_position_id_with_mask(attention_mask).tolist() == [0, 0, 1, 2, 3, 4, 5, 5]


def test_prompt_is_truncated_before_response_when_needed() -> None:
    source = _Tokenizer()
    target = _Tokenizer(prompt_ids=(80, 81, 82, 83, 84))
    input_ids, attention_mask, responses, valid_length = _build_token_level_teacher_inputs(
        raw_prompt=[{"role": "user", "content": "question"}],
        source_response_ids=torch.tensor([10, 11, 12]),
        response_mask=torch.tensor([1, 1, 1]),
        source_tokenizer=source,
        target_tokenizer=target,
        response_length=3,
        max_length=6,
    )

    assert valid_length == 3
    assert input_ids.tolist() == [82, 83, 84, 10, 11, 12]
    assert attention_mask.tolist() == [1] * 6
    assert torch.equal(input_ids[-valid_length:], responses[:valid_length])


def test_mismatched_eos_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="special-token IDs"):
        _validate_token_level_tokenizers(_Tokenizer(eos_token_id=2), _Tokenizer(eos_token_id=3))


def test_mismatched_token_id_mapping_fails_closed() -> None:
    with pytest.raises(ValueError, match="token-to-ID mappings"):
        _validate_token_level_tokenizers(_Tokenizer(), _Tokenizer(vocab_shift=1))


def _build_row(valid_ids, *, width=6, total=10, prompt=(90, 91, 92)):
    return _build_token_level_teacher_inputs(
        raw_prompt=[{"role": "user", "content": "question"}],
        source_response_ids=torch.tensor(list(valid_ids) + [0] * (width - len(valid_ids))),
        response_mask=torch.tensor([1] * len(valid_ids) + [0] * (width - len(valid_ids))),
        source_tokenizer=_Tokenizer(), target_tokenizer=_Tokenizer(prompt_ids=prompt),
        response_length=width, max_length=total,
    )


@pytest.fixture
def cpu_packing(monkeypatch):
    """Replace only hardware padding kernels; run the production packing/extraction code."""
    import einops
    import verl.utils.attention_utils as attention

    def unpad(values, mask):
        indices = mask.flatten().nonzero().flatten()
        return values.flatten(0, 1)[indices], indices, None, None

    def pad(hidden_states, indices, batch, seqlen):
        full = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
        full[indices] = hidden_states
        return full.view(batch, seqlen, *hidden_states.shape[1:])

    monkeypatch.setattr(attention, "unpad_input", unpad)
    monkeypatch.setattr(attention, "pad_input", pad)
    monkeypatch.setattr(attention, "index_first_axis", lambda values, indices: values[indices])
    monkeypatch.setattr(attention, "rearrange", einops.rearrange)


@pytest.mark.parametrize("remove_padding", [False, True])
def test_mixed_lengths_eos_full_width_and_row_reordering(cpu_packing, remove_padding):
    responses = [[10, 2], [20, 21, 22, 23, 24, 2], [30, 31, 2]]
    rows = [_build_row(ids) for ids in responses]
    scores, candidates = _native_scores(rows, remove_padding=remove_padding)
    for row, valid_ids in enumerate(responses):
        _assert_native_reference(row, scores, candidates, prompt_ids=[90, 91, 92], valid_ids=valid_ids)
    permutation = [2, 0, 1]
    reordered, _ = _native_scores(
        [rows[i] for i in permutation], remove_padding=remove_padding, candidates=candidates[permutation]
    )
    for actual, original in zip(reordered, scores, strict=True):
        torch.testing.assert_close(actual, original[permutation])


def test_if_cap_inside_mixed_batch_response_width():
    valid_ids = [10] * 2047 + [2]
    row = _build_row(valid_ids, width=16384, total=18432, prompt=tuple(range(100)) * 21)
    assert row[0].shape == (18432,)
    assert row[0][-16384:].tolist() == valid_ids + [0] * (16384 - 2048)
    # Independent locked policy: keep only the final P=2048 prompt IDs, even for short IF responses.
    prompt_ids = (list(range(100)) * 21)[-2048:]
    scores, candidates = _native_scores([row])
    _assert_native_reference(0, scores, candidates, prompt_ids=prompt_ids, valid_ids=valid_ids)


@pytest.mark.parametrize("layout", ["legacy", "shifted"])
def test_wrong_positions_are_rejected_by_native_reference(layout):
    row = list(_build_row([10, 11]))
    if layout == "legacy":
        row[0] = torch.tensor([0, 0, 0, 0, 0, 90, 91, 92, 10, 11])
        row[1] = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    else:
        row[0] = row[0].roll(1)
        row[1] = row[1].roll(1)
    scores, candidates = _native_scores([row])
    with pytest.raises(AssertionError):
        _assert_native_reference(0, scores, candidates, prompt_ids=[90, 91, 92], valid_ids=[10, 11])


def test_masked_teacher_positions_do_not_reach_policy_gradient():
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    row = _build_row([10, 2])
    scores, _ = _native_scores([row])
    current = torch.full_like(scores[1], -5.0, requires_grad=True)
    old = current.detach().clone()
    advantages = (scores[1] - current.detach()) * current.detach().exp()
    mask = torch.tensor([[1, 1, 0, 0, 0, 0]])
    config = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.28})
    loss, _ = compute_policy_loss_vanilla(old, current, advantages, mask, config=config)
    loss.backward()
    assert current.grad[:, :2].abs().sum() > 0
    assert torch.count_nonzero(current.grad[:, 2:]) == 0


def test_builder_does_not_repeat_initialization_vocabulary_scan(monkeypatch):
    source, target = _Tokenizer(), _Tokenizer()
    _validate_token_level_tokenizers(source, target)
    def unexpected_scan():
        pytest.fail("per-sample builder traversed tokenizer vocabulary")
    monkeypatch.setattr(source, "get_vocab", unexpected_scan)
    monkeypatch.setattr(target, "get_vocab", unexpected_scan)
    _build_token_level_teacher_inputs(
        raw_prompt=[{"role": "user", "content": "question"}],
        source_response_ids=torch.tensor([10, 2]), response_mask=torch.ones(2),
        source_tokenizer=source, target_tokenizer=target, response_length=2, max_length=6,
    )


def test_tokenizer_mismatch_is_rejected_at_model_initialization(monkeypatch):
    from omegaconf import OmegaConf
    import verl.workers.fsdp_workers as workers

    config = OmegaConf.create({"model": {"path": "teacher", "input_tokenizer": "student"}})
    worker = object.__new__(RewardModelWorker)
    worker.config = config
    monkeypatch.setattr(workers, "copy_to_local", lambda path, **kwargs: path)
    tokenizers = {"student": _Tokenizer(), "teacher": _Tokenizer(eos_token_id=3)}
    monkeypatch.setattr(workers, "hf_tokenizer", lambda path, **kwargs: tokenizers[path])
    with pytest.raises(ValueError, match="special-token IDs"):
        worker._build_model(config)


def test_native_template_switch_keeps_fixed_width_and_position_ids():
    import numpy as np
    from omegaconf import OmegaConf
    from verl import DataProto

    worker = object.__new__(RewardModelWorker)
    worker.config = OmegaConf.create({"max_length": None})
    worker.input_tokenizer = _Tokenizer()
    worker.tokenizer = _Tokenizer()
    worker._rank = 1
    prompt = np.empty(1, dtype=object)
    prompt[0] = [{"role": "user", "content": "question"}]
    data = DataProto.from_dict(
        tensors={
            "attention_mask": torch.ones(1, 10),
            "responses": torch.tensor([[10, 2, 0, 0, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0, 0, 0, 0]]),
        },
        non_tensors={"raw_prompt": prompt},
    )
    output = worker._switch_chat_template_token_level(data).batch
    assert output["input_ids"].tolist() == [[0, 90, 91, 92, 10, 2, 0, 0, 0, 0]]
    assert output["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
    assert output["position_ids"].tolist() == [[0, 0, 1, 2, 3, 4, 4, 4, 4, 4]]
    assert torch.equal(output["responses"], data.batch["responses"])


@pytest.mark.parametrize("mask", [[1, 0, 1], [1, 2, 0]])
def test_non_prefix_response_mask_is_rejected(mask):
    with pytest.raises(ValueError, match="valid prefix"):
        _build_token_level_teacher_inputs(
            raw_prompt=[{"role": "user", "content": "question"}],
            source_response_ids=torch.tensor([10, 11, 0]), response_mask=torch.tensor(mask),
            source_tokenizer=_Tokenizer(), target_tokenizer=_Tokenizer(), response_length=3, max_length=7,
        )
