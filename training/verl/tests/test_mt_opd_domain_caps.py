"""Per-domain generation caps must apply before sampling, not after truncation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto

from verl.workers.actor.mt_opd import PAPER_DOMAIN_RESPONSE_CAPS, domain_generation_groups
from verl.workers.config.rollout import RolloutConfig


def test_if_uses_2048_and_math_code_use_16384() -> None:
    domains = ["if", "math", "code", "if", "math"]
    groups = domain_generation_groups(domains, PAPER_DOMAIN_RESPONSE_CAPS)
    by_domain = {name: (indices, cap) for name, indices, cap in groups}
    assert by_domain["if"] == ([0, 3], 2048)
    assert by_domain["math"] == ([1, 4], 16384)
    assert by_domain["code"] == ([2], 16384)


def test_unknown_domain_fails_before_generation() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        domain_generation_groups(["math", "other"], PAPER_DOMAIN_RESPONSE_CAPS)


def test_rollout_config_accepts_domain_response_length() -> None:
    cfg = RolloutConfig(
        name="vllm",
        domain_response_length=dict(PAPER_DOMAIN_RESPONSE_CAPS),
    )
    assert cfg.get("domain_response_length")["if"] == 2048
    assert cfg.domain_response_length["math"] == 16384


def test_restore_order_is_the_original_row_index() -> None:
    domains = ["code", "if", "math"]
    groups = domain_generation_groups(domains, PAPER_DOMAIN_RESPONSE_CAPS)
    restored = [None] * 3
    fake_outputs = {"code": ["C"], "if": ["I"], "math": ["M"]}
    for name, indices, _cap in groups:
        for local, orig in enumerate(indices):
            restored[orig] = fake_outputs[name][local]
    assert restored == ["C", "I", "M"]


@pytest.mark.parametrize("pad_id", [0, 2])
@pytest.mark.parametrize("if_has_eos", [False, True])
def test_native_rollout_masks_only_generated_tokens_after_domain_padding(pad_id, if_has_eos):
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

    domains = ["code", "if", "math", "if"]
    generated = [[10] * 16384, [11] * 2048, [12, 13, 2], [14, 15, 16]]
    if if_has_eos:
        generated[1][-1] = 2
    calls = []

    class Engine:
        def generate(self, *, prompts, sampling_params, **kwargs):
            rows = [item["prompt_token_ids"][0] - 90 for item in prompts]
            calls.append((rows, sampling_params.max_tokens))
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=generated[row])]) for row in rows]

    worker = object.__new__(vLLMRollout)
    worker.config = OmegaConf.create({
        "response_length": 16384, "domain_response_length": PAPER_DOMAIN_RESPONSE_CAPS,
        "calculate_log_probs": False,
    })
    worker.pad_token_id = pad_id
    worker.lora_kwargs = {}
    worker.sampling_params = SimpleNamespace(max_tokens=16384)
    worker.inference_engine = Engine()
    ids = torch.tensor([[90 + row, 80, 81] for row in range(4)])
    prompts = DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "position_ids": torch.arange(3).expand(4, -1)},
        non_tensors={"domain": np.array(domains)}, meta_info={"eos_token_id": 2},
    )
    output = worker.generate_sequences(prompts)
    mask = output.batch["attention_mask"][:, -16384:]
    assert calls == [([0], 16384), ([1, 3], 2048), ([2], 16384)]
    assert worker.sampling_params.max_tokens == 16384
    assert mask.sum(-1).tolist() == [16384, 2048, 3, 3]
    assert output.batch["responses"].shape == (4, 16384)
    assert output.batch["input_ids"].shape == (4, 16387)
    for row, tokens in enumerate(generated):
        assert output.batch["responses"][row, :len(tokens)].tolist() == tokens
        assert mask[row, :len(tokens)].eq(1).all()
        assert mask[row, len(tokens):].eq(0).all()
        assert output.batch["responses"][row, len(tokens):].eq(pad_id).all()
