"""M4 routed-q lifecycle: keep canonical q until refresh consumes it.

The previous trainer cleanup always deleted ``teacher_on_student_log_probs``
after advantage, so the learner never saw the routed teacher scores. Extra
``mt_teacher_*`` tensors are diagnostics after routing; they are not a
substitute for the canonical field.
"""
from __future__ import annotations

import pytest
import torch

from verl.workers.actor.mt_opd import (
    ROUTED_TEACHER_Q_KEY,
    batch_keys_to_drop_before_actor_update,
    build_domain_weights,
    drop_unused_teacher_tensors_before_actor_update,
    refresh_opd_advantage,
    require_m4_refresh_tensors,
    select_routed_teacher_logprobs,
)


DOMAIN_ORDER = ["math", "code", "if"]


def _three_domain_teachers(batch_size: int = 6, seq_len: int = 3, top_k: int = 4):
    domains = ["math", "code", "if", "math", "code", "if"][:batch_size]
    teachers = []
    for teacher_idx in range(3):
        scores = torch.empty(batch_size, seq_len, top_k)
        for row in range(batch_size):
            scores[row] = float((teacher_idx + 1) * 100 + row)
        teachers.append(scores)
    ids = torch.arange(batch_size * seq_len * top_k, dtype=torch.long).reshape(
        batch_size, seq_len, top_k
    )
    mask = torch.ones(batch_size, seq_len)
    mask[:, -1] = 0
    return domains, teachers, ids, mask


def _route(domains, teachers):
    weights = build_domain_weights(
        domains, domain_order=DOMAIN_ORDER, device="cpu"
    )
    return select_routed_teacher_logprobs(teachers, weights)


def test_unknown_domain_fails_instead_of_uniform_fallback() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        build_domain_weights(
            ["math", "other"], domain_order=DOMAIN_ORDER, device="cpu"
        )
    with pytest.raises(ValueError, match="per-sample domain labels"):
        build_domain_weights(None, domain_order=DOMAIN_ORDER, device="cpu")


def test_routing_picks_the_matching_teacher_not_the_first_extra() -> None:
    domains, teachers, _, _ = _three_domain_teachers()
    routed = _route(domains, teachers)
    math_q, code_q, if_q = teachers
    assert not torch.equal(routed, math_q)
    assert not torch.equal(routed, code_q)
    assert not torch.equal(routed, if_q)
    for row, domain in enumerate(domains):
        expected = {"math": math_q, "code": code_q, "if": if_q}[domain]
        torch.testing.assert_close(routed[row], expected[row])


def test_refresh_keeps_routed_q_and_drops_diagnostics() -> None:
    domains, teachers, ids, mask = _three_domain_teachers()
    routed = _route(domains, teachers)
    batch = {
        ROUTED_TEACHER_Q_KEY: routed.clone(),
        "student_top_k_ids": ids,
        "response_mask": mask,
        "mt_teacher_1_on_student_log_probs": teachers[1].clone(),
        "mt_teacher_2_on_student_log_probs": teachers[2].clone(),
        "teacher_entropy": torch.ones_like(mask),
        "teacher_top_k_ids": ids.clone(),
        "advantages": torch.zeros_like(routed),
    }

    drop_unused_teacher_tensors_before_actor_update(batch, refresh_advantage=True)
    require_m4_refresh_tensors(batch.keys(), refresh_advantage=True)

    assert ROUTED_TEACHER_Q_KEY in batch
    torch.testing.assert_close(batch[ROUTED_TEACHER_Q_KEY], routed)
    assert "mt_teacher_1_on_student_log_probs" not in batch
    assert "mt_teacher_2_on_student_log_probs" not in batch
    assert "teacher_entropy" not in batch
    assert "student_top_k_ids" in batch
    assert "advantages" in batch


def test_refresh_off_still_drops_canonical_q() -> None:
    batch = {
        ROUTED_TEACHER_Q_KEY: torch.zeros(2, 2, 2),
        "student_top_k_ids": torch.zeros(2, 2, 2, dtype=torch.long),
        "response_mask": torch.ones(2, 2),
        "teacher_entropy": torch.ones(2, 2),
    }
    drop_unused_teacher_tensors_before_actor_update(batch, refresh_advantage=False)
    require_m4_refresh_tensors(batch.keys(), refresh_advantage=False)
    assert ROUTED_TEACHER_Q_KEY not in batch
    assert "teacher_entropy" not in batch


def test_missing_routed_q_fails_even_if_mt_teacher_keys_remain() -> None:
    keys = [
        "mt_teacher_1_on_student_log_probs",
        "mt_teacher_2_on_student_log_probs",
        "student_top_k_ids",
        "response_mask",
    ]
    with pytest.raises(ValueError, match="not a substitute"):
        require_m4_refresh_tensors(keys, refresh_advantage=True)


def test_missing_support_ids_fails_when_refresh_is_on() -> None:
    keys = [ROUTED_TEACHER_Q_KEY, "response_mask"]
    with pytest.raises(ValueError, match="student_top_k_ids"):
        require_m4_refresh_tensors(keys, refresh_advantage=True)


def test_routed_q_stays_aligned_after_shuffle_and_microbatch() -> None:
    domains, teachers, ids, mask = _three_domain_teachers()
    routed = _route(domains, teachers)
    perm = [5, 0, 3, 1, 4, 2]
    q_shuf = routed[perm]
    ids_shuf = ids[perm]
    mask_shuf = mask[perm]
    domains_shuf = [domains[i] for i in perm]
    orig_rows = perm

    batch = {
        ROUTED_TEACHER_Q_KEY: q_shuf.clone(),
        "student_top_k_ids": ids_shuf.clone(),
        "response_mask": mask_shuf.clone(),
        "mt_teacher_1_on_student_log_probs": teachers[1][perm].clone(),
    }
    drop_unused_teacher_tensors_before_actor_update(batch, refresh_advantage=True)
    require_m4_refresh_tensors(batch.keys(), refresh_advantage=True)

    micro_size = 2
    for start in range(0, len(perm), micro_size):
        sl = slice(start, start + micro_size)
        micro_q = batch[ROUTED_TEACHER_Q_KEY][sl]
        for local, (domain, orig) in enumerate(
            zip(domains_shuf[sl], orig_rows[sl], strict=True)
        ):
            expected = teachers[DOMAIN_ORDER.index(domain)][orig]
            torch.testing.assert_close(micro_q[local], expected)
            torch.testing.assert_close(
                batch["student_top_k_ids"][start + local], ids[orig]
            )


def test_refresh_matches_student_p_formula_and_detaches_student() -> None:
    torch.manual_seed(0)
    student = torch.randn(3, 4, 5, requires_grad=True)
    teacher = torch.randn(3, 4, 5)
    mask = torch.ones(3, 4)
    mask[0, -1] = 0
    got = refresh_opd_advantage(
        student, teacher, mask, reward_weight_mode="student_p"
    )
    weights = torch.softmax(student.detach(), dim=-1)
    expected = (teacher.to(student.dtype) - student.detach()) * weights
    expected = expected * mask.unsqueeze(-1)
    torch.testing.assert_close(got, expected)
    assert not got.requires_grad


def test_drop_list_never_aliases_mt_teacher_onto_canonical() -> None:
    keys = [
        ROUTED_TEACHER_Q_KEY,
        "mt_teacher_1_on_student_log_probs",
        "student_top_k_ids",
        "response_mask",
    ]
    drop_on = batch_keys_to_drop_before_actor_update(keys, refresh_advantage=True)
    assert ROUTED_TEACHER_Q_KEY not in drop_on
    assert "mt_teacher_1_on_student_log_probs" in drop_on
    drop_off = batch_keys_to_drop_before_actor_update(keys, refresh_advantage=False)
    assert ROUTED_TEACHER_Q_KEY in drop_off
