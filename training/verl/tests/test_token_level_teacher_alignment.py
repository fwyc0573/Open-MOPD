"""Regression tests for explicit teacher response-position alignment."""

import pytest
import torch

from verl.workers.fsdp_workers import _build_token_level_teacher_inputs


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


def test_teacher_sequence_ends_with_the_sampled_response_ids() -> None:
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
    assert attention_mask.tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    assert responses.tolist() == [10, 11, 12, 0]
    assert torch.equal(input_ids[-valid_length:], responses[:valid_length])


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
        _build_token_level_teacher_inputs(
            raw_prompt=[{"role": "user", "content": "question"}],
            source_response_ids=torch.tensor([10]),
            response_mask=torch.tensor([1]),
            source_tokenizer=_Tokenizer(eos_token_id=2),
            target_tokenizer=_Tokenizer(eos_token_id=3),
            response_length=1,
            max_length=4,
        )


def test_mismatched_token_id_mapping_fails_closed() -> None:
    with pytest.raises(ValueError, match="token-to-ID mappings"):
        _build_token_level_teacher_inputs(
            raw_prompt=[{"role": "user", "content": "question"}],
            source_response_ids=torch.tensor([10]),
            response_mask=torch.tensor([1]),
            source_tokenizer=_Tokenizer(),
            target_tokenizer=_Tokenizer(vocab_shift=1),
            response_length=1,
            max_length=4,
        )
