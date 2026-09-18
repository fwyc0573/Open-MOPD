"""Per-domain generation caps must apply before sampling, not after truncation."""

import pytest

from verl.workers.actor.mt_opd import PAPER_DOMAIN_RESPONSE_CAPS, domain_generation_groups


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


def test_restore_order_is_the_original_row_index() -> None:
    domains = ["code", "if", "math"]
    groups = domain_generation_groups(domains, PAPER_DOMAIN_RESPONSE_CAPS)
    restored = [None] * 3
    fake_outputs = {"code": ["C"], "if": ["I"], "math": ["M"]}
    for name, indices, _cap in groups:
        for local, orig in enumerate(indices):
            restored[orig] = fake_outputs[name][local]
    assert restored == ["C", "I", "M"]
