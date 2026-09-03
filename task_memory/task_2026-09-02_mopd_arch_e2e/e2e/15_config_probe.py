#!/usr/bin/env python3
"""Open-MOPD E2E — Step 15: compose-only Hydra probe for every stage's override set.

Why this step exists. A wrong Hydra key does not fail fast on a GPU: it fails after
Ray has booted, vLLM has allocated KV cache, and four FSDP models have loaded. This
probe composes the exact same override lists with hydra.compose() and no execution,
so a key error costs a second instead of ten minutes of H800 time.

It also verifies the launcher contract: reward_mode and log_prob_top_k are runtime
extension keys, so the local launchers must append them with a leading '+'.

Usage:
    python 15_config_probe.py [--assets DIR]

Exit code 0 only when every case lands on its expected verdict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path("/data/ycfeng/Open-MOPD")
PPO_CONFIG_DIR = REPO / "training/verl/verl/trainer/config"


def compose(config_name: str, overrides: list[str]):
    """Return (ok, detail). Isolated per call so Hydra's global state stays clean."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(PPO_CONFIG_DIR), version_base=None):
        try:
            cfg = hydra_compose(config_name=config_name, overrides=overrides)
        except Exception as exc:  # noqa: BLE001 — the exception class is the finding
            return False, f"{type(exc).__name__}: {exc}"
        return True, cfg


def common_ppo(a: Path, gpus: int = 1) -> list[str]:
    p, r = 128, 64
    return [
        f"data.train_files={a}/data/rl_train.parquet",
        f"data.val_files={a}/data/rl_val.parquet",
        "data.train_batch_size=12",
        f"data.max_prompt_length={p}",
        f"data.max_response_length={r}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={a}/models/student",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=eager",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size=12",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.optim.lr=1e-4",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.25",
        f"actor_rollout_ref.rollout.max_model_len={p + r}",
        "actor_rollout_ref.rollout.max_num_batched_tokens=1024",
        "actor_rollout_ref.rollout.max_num_seqs=64",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        f"trainer.n_gpus_per_node={gpus}",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=2",
        "trainer.logger=['console']",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=2",
        "trainer.resume_mode=disable",
    ]


def teacher(i: int, path: Path) -> list[str]:
    return [
        f"+mt_reward_model_{i}.enable=True",
        f"+mt_reward_model_{i}.model.path={path}",
        f"+mt_reward_model_{i}.model.input_tokenizer=null",
        f"+mt_reward_model_{i}.model.use_remove_padding=False",
        f"+mt_reward_model_{i}.model.override_config.attn_implementation=eager",
        f"+mt_reward_model_{i}.micro_batch_size_per_gpu=2",
    ]


def build_cases(a: Path):
    val_reward = REPO / "training/verl/verl/utils/reward_score/opd_val_dispatch.py"
    t_math, t_code, t_if = (a / "models" / n for n in ("teacher_math", "teacher_code", "teacher_if"))

    rm_common = [
        "reward_model.enable=True",
        "reward_model.strategy=fsdp",
        "reward_model.model.input_tokenizer=null",
        "reward_model.model.use_remove_padding=False",
        "+reward_model.model.override_config.attn_implementation=eager",
        "reward_model.micro_batch_size_per_gpu=2",
        f"custom_reward_function.path={val_reward}",
        "custom_reward_function.name=reward_func",
    ]

    cases = []

    # --- baseline: does ppo_trainer compose at all
    cases.append(("ppo baseline", "ppo_trainer", ["actor_rollout_ref.rollout.name=vllm"], True))

    # --- SFT
    cases.append((
        "sft (as scripts/local/sft.sh emits it)",
        "sft_trainer_engine",
        [
            f"data.train_files={a}/data/sft_train.parquet",
            f"data.val_files={a}/data/sft_val.parquet",
            "data.train_batch_size=4",
            "data.max_length=256",
            "data.truncation=error",
            "data.use_dynamic_bsz=True",
            "data.micro_batch_size_per_gpu=2",
            "data.max_token_len_per_gpu=1024",
            "+model.override_config.attn_implementation=eager",
            f"model.path={a}/models/student",
            "optim.lr=1e-4",
            "trainer.project_name=OpenMOPD-e2e",
            "trainer.experiment_name=sft-dummy",
            "trainer.total_epochs=1",
            f"trainer.default_local_dir={a}/../runs/sft/checkpoints",
            "trainer.resume_mode=disable",
            "trainer.logger=['console']",
            "trainer.save_freq=1",
            "trainer.test_freq=1",
        ],
        True,
    ))
    # The key sft.sh does NOT emit but sft_trainer.yaml (the OTHER trainer) declares.
    cases.append((
        "sft with prompt_key/response_key (wrong trainer's keys)",
        "sft_trainer_engine",
        ["data.prompt_key=question", "data.response_key=answer"],
        False,
    ))

    # --- RL (GRPO)
    cases.append((
        "rl grpo (as scripts/local/rl.sh emits it)",
        "ppo_trainer",
        ["algorithm.adv_estimator=grpo", *common_ppo(a), "reward_model.enable=False",
         f"custom_reward_function.path={val_reward}", "custom_reward_function.name=reward_func"],
        True,
    ))

    # --- OPD: retain the invalid plain form as a negative regression and verify the launcher form.
    cases.append((
        "opd invalid plain reward_mode (negative regression)",
        "ppo_trainer",
        ["algorithm.adv_estimator=token_reward_direct", *common_ppo(a),
         "actor_rollout_ref.rollout.reward_mode=opd_kl", *rm_common,
         f"reward_model.model.path={t_math}"],
        False,  # expected to FAIL — this is the launcher defect
    ))
    cases.append((
        "opd launcher form (+reward_mode, +log_prob_top_k)",
        "ppo_trainer",
        ["algorithm.adv_estimator=token_reward_direct", *common_ppo(a),
         "+actor_rollout_ref.rollout.reward_mode=opd_kl",
         "+actor_rollout_ref.rollout.log_prob_top_k=16",
         *rm_common, f"reward_model.model.path={t_math}"],
        True,
    ))

    # --- MT-OPD: retain the invalid plain form as a negative regression and verify the launcher form.
    mt_invalid = [
        "algorithm.adv_estimator=token_reward_direct", *common_ppo(a),
        "actor_rollout_ref.rollout.reward_mode=mt_opd",
        "reward_model.enable=True", f"reward_model.model.path={t_math}",
        "+mt_opd.teacher_domains=[math,code,if]",
        "+mt_opd.n_additional_teachers=2",
        *teacher(1, t_code), *teacher(2, t_if),
    ]
    cases.append(("mt_opd invalid plain reward_mode (negative regression)", "ppo_trainer", mt_invalid, False))

    mt_fixed = [
        "algorithm.adv_estimator=token_reward_direct", *common_ppo(a),
        "+actor_rollout_ref.rollout.reward_mode=mt_opd",
        "+actor_rollout_ref.rollout.log_prob_top_k=16",
        *rm_common, f"reward_model.model.path={t_math}",
        "+mt_opd.teacher_domains=[math,code,if]",
        "+mt_opd.n_additional_teachers=2",
        "+mt_opd.domain_weighting=domain_routing",
        *teacher(1, t_code), *teacher(2, t_if),
    ]
    cases.append(("mt_opd CORRECTED (naive M-OPD: routing only)", "ppo_trainer", mt_fixed, True))

    cases.append((
        "mt_opd CORRECTED + M1 target shares",
        "ppo_trainer",
        [*mt_fixed,
         "+mt_opd.target_share_domains=[math,code,if]",
         "+mt_opd.target_share_values=[0.4,0.4,0.2]"],
        True,
    ))
    cases.append((
        "mt_opd CORRECTED + M1 + M2 + M3",
        "ppo_trainer",
        [*mt_fixed,
         "+mt_opd.target_share_domains=[math,code,if]",
         "+mt_opd.target_share_values=[0.4,0.4,0.2]",
         "+mt_opd.normalize_reward_scale=1.0",
         "+mt_opd.reward_scale_stat=mean",
         "+mt_opd.reward_scale_direction=multiply",
         "+mt_opd.conflict_policy=mask",
         "+mt_opd.conflict_nats=1.0"],
        True,
    ))
    # target_gradient_shares as a dict: the source comments say this form does not compose.
    cases.append((
        "mt_opd + target_gradient_shares as a dict literal",
        "ppo_trainer",
        [*mt_fixed, "+mt_opd.target_gradient_shares={math:0.4,code:0.4,if:0.2}"],
        None,  # unknown expectation — this probe settles it
    ))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=Path("/data/ycfeng/tmp/openmopd-e2e/assets"))
    args = ap.parse_args()

    cases = build_cases(args.assets)
    failures = 0
    print(f"{'':2s} {'case':52s} {'expected':9s} {'actual':9s}")
    print("-" * 90)
    details = []
    for name, cfg_name, overrides, expected in cases:
        ok, detail = compose(cfg_name, overrides)
        exp = {True: "COMPOSE", False: "RAISE", None: "unknown"}[expected]
        act = "COMPOSE" if ok else "RAISE"
        verdict = "  "
        if expected is not None and (ok is not expected):
            verdict = "!!"
            failures += 1
        print(f"{verdict} {name:52s} {exp:9s} {act:9s}")
        if not ok:
            details.append((name, str(detail).strip().splitlines()[0][:200]))

    if details:
        print("\n--- RAISE details ---")
        for name, msg in details:
            print(f"  {name}\n      {msg}")

    print(f"\nunexpected outcomes: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
