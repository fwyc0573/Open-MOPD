"""Gap direction multiply: paper Eq.9 vs author_native share-weighted m_ref."""

import pytest
import torch

from verl.workers.actor.mt_opd import (
    compute_domain_loss_weights,
    gap_reference_magnitude,
)


def test_nonuniform_share_and_clamp_split_the_two_references() -> None:
    """Fixture from v4 RC-04: shares [.98,.01,.01] and magnitudes [1e-6, 1, 100]."""
    magnitudes = {"math": 1e-6, "code": 1.0, "if": 100.0}
    shares = {"math": 0.98, "code": 0.01, "if": 0.01}
    paper = gap_reference_magnitude(magnitudes, shares, "paper_equation")
    native = gap_reference_magnitude(magnitudes, shares, "author_native")
    assert paper == pytest.approx((1e-6 + 1.0 + 100.0) / 3.0)
    assert native == pytest.approx(0.98 * 1e-6 + 0.01 * 1.0 + 0.01 * 100.0)
    assert paper != pytest.approx(native)
    math_fac_paper = min(max(1e-6 / paper, 0.05), 20.0)
    math_fac_native = min(max(1e-6 / native, 0.05), 20.0)
    if_fac_paper = min(max(100.0 / paper, 0.05), 20.0)
    if_fac_native = min(max(100.0 / native, 0.05), 20.0)
    assert math_fac_paper == pytest.approx(0.05)
    assert math_fac_native == pytest.approx(0.05)
    assert if_fac_native == pytest.approx(20.0)
    assert if_fac_paper < 20.0
    assert if_fac_paper != pytest.approx(if_fac_native)


def test_author_native_multiply_is_the_current_helper() -> None:
    domains = ["math"] * 98 + ["code"] + ["if"]
    mask = torch.ones(100, 1)
    rm = torch.zeros(100, 1)
    rm[:98] = 1e-6
    rm[98] = 1.0
    rm[99] = 100.0
    weights = compute_domain_loss_weights(
        domains,
        mask,
        {"math": 1 / 3, "code": 1 / 3, "if": 1 / 3},
        rm_scores=rm,
        normalize_reward_scale=1.0,
        reward_scale_direction="multiply",
    )
    assert weights is not None
    assert weights[99].item() > weights[0].item()
